import sys
import threading
from collections import deque
from typing import Dict

import click
import cv2
import dill
import hydra
import numpy as np
import pygame
import torch
from omegaconf import OmegaConf
from pygame.locals import *

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from shepherd_game.game import BLACK, Game
from shepherd_game.parameters import FIELD_LENGTH, PADDING
from shepherd_game.utils import dist

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

OmegaConf.register_new_resolver("eval", eval, replace=True)


class action_predictor:
    def __init__(self,
                 ckpt_path: str,
                 obs: Dict[str, any],
                 device: str = 'cpu',
                 seed: int = None,
                 save_path: str = None,
                 param_dict: Dict[str, any] = None,
                 label: str = None,
                 model_num_sheep: int = 1,
                 model_num_dog: int = 2) -> None:

        self.video: cv2.VideoWriter | None = None
        if save_path:
            if '.mp4' not in save_path:
                save_path = save_path + '.mp4'
            width = (FIELD_LENGTH + 2 * PADDING[0]) * 4
            height = (FIELD_LENGTH + 2 * PADDING[1]) * 4
            self.video = cv2.VideoWriter(
                save_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                15, (width, height))

        if param_dict is None:
            param_dict = {}
        param_dict['num_dog'] = model_num_dog
        param_dict.setdefault('num_sheep', 5)
        self.env = Game(seed=seed, **param_dict)

        self.fpsClock = pygame.time.Clock()

        self.payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
        self.cfg = self.payload['cfg']

        workspace_cls = hydra.utils.get_class(self.cfg._target_)
        self.workspace: BaseWorkspace = workspace_cls(self.cfg)
        self.workspace.load_payload(self.payload)

        self.policy: BaseImagePolicy = self.workspace.model
        if self.cfg.training.use_ema:
            self.policy = self.workspace.ema_model

        self.device = device
        self.policy.eval().to(self.device)

        self.obs_deque = deque(maxlen=self.policy.n_obs_steps)

        if model_num_sheep == self.env.num_agents:
            def update_func():
                return np.array([pos for sheep in self.env.sheep for pos in sheep]).astype(np.float32)
        else:
            def update_func():
                return np.mean(self.env.sheep, axis=0).astype(np.float32)

        self.obs = dict.fromkeys(obs.keys())
        self.obs_updater = {
            'image': lambda: np.transpose(pygame.surfarray.array3d(self.env.screen.convert()), (2, 1, 0)).astype(np.float32),
            'pos': None,
            'sheep_pos': update_func,
            'sheep_com': lambda: np.mean(self.env.sheep, axis=0).astype(np.float32),
            'goal': lambda: self.env.target.astype(np.float32),
        }

        self.action = []
        self.label = label

    def choose_dog_to_control(self) -> int:
        sheep_com = self.env.sheep.mean(axis=0)
        target = self.env.target
        v_line = sheep_com - target

        dogs = self.env.dog
        cross_values = []

        for i in range(len(dogs)):
            dog_vec = dogs[i] - target
            cross = v_line[0] * dog_vec[1] - v_line[1] * dog_vec[0]
            cross_values.append(cross)

        if cross_values[0] > 0 and cross_values[1] <= 0:
            return 0
        elif cross_values[1] > 0 and cross_values[0] <= 0:
            return 1
        else:
            dist0 = np.linalg.norm(self.env.dog[0] - sheep_com)
            dist1 = np.linalg.norm(self.env.dog[1] - sheep_com)
            return 0 if dist0 < dist1 else 1

    def get_obs(self, active_dog_idx: int) -> Dict[str, np.ndarray]:
        for key, update_func in self.obs_updater.items():
            if key == 'pos':
                self.obs[key] = self.env.dog[active_dog_idx].astype(np.float32)
            elif key in self.obs:
                self.obs[key] = update_func()

        return {
            'obs': np.hstack([
                self.obs['pos'],
                self.obs['sheep_pos'],
                self.obs['sheep_com'],
                self.obs['goal']
            ])
        }

    def infer(self) -> None:
        obs_data_tensors = dict_apply({
            key: np.stack([data[key] for data in self.obs_deque])
            for key in self.obs_deque[0].keys()},
            lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))

        with torch.no_grad():
            result = self.policy.predict_action(obs_data_tensors)

        if len(self.action) == 0:
            self.action = result['action'].squeeze(0).to('cpu').numpy().tolist()

    def run(self, iters: int = 60 * 15) -> None:

        active_idx = 0
        obs = self.get_obs(active_idx)
        self.obs_deque.append(obs)
        self.obs_deque.append(obs)

        inference_thread = threading.Thread(target=self.infer)
        inference_thread.start()

        count = 0
        while count < iters:

            self.env.render()


            com = np.mean(np.array([pos for sheep in self.env.sheep for pos in sheep]).reshape(-1, 2), axis=0)
            goal = self.env.target
            v_main = goal - com

            dogA = self.env.dog[0]
            dogB = self.env.dog[1]
            vecA = com - dogA
            vecB = com - dogB

            crossA = v_main[0] * vecA[1] - v_main[1] * vecA[0]
            crossB = v_main[0] * vecB[1] - v_main[1] * vecB[0]

            if crossA > 0 and crossB <= 0:
                active_idx = 0
            elif crossB > 0 and crossA <= 0:
                active_idx = 1
            else:
                active_idx = 0 if np.linalg.norm(vecA) < np.linalg.norm(vecB) else 1

            passive_idx = 1 - active_idx


            obs = self.get_obs(active_idx)
            self.obs_deque.append(obs)

            if len(self.action) > 0:
                dog_action = self.action.pop()


                full_action = np.zeros_like(self.env.dog)
                full_action[active_idx] = dog_action

                vec = com - self.env.dog[passive_idx]
                dist = np.linalg.norm(vec)
                if dist > 1e-2:
                    vec = vec / (dist + 1e-6) * 0.6
                else:
                    vec = np.zeros(2)
                full_action[passive_idx] = vec

                #print(f"[Frame {count}] Active dog: {active_idx}, Action: {np.round(dog_action, 3)}")

                done = self.env.step(full_action)
            else:
                #print(f"[Frame {count}] Waiting for model action...")
                self.fpsClock.tick(25)
                continue


            if self.video is not None:
                frame = np.transpose(pygame.surfarray.array3d(self.env.screen), (1, 0, 2))
                frame = frame[..., ::-1]
                scaled = np.repeat(np.repeat(frame, 4, axis=0), 4, axis=1)
                scaled = np.clip(scaled, 0, 255).astype(np.uint8)
                if self.label:
                    cv2.putText(scaled, self.label, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 2, BLACK, 2)
                self.video.write(scaled)

            if done:
                #print("✅ Success!")
                while count < iters:
                    if self.video is not None:
                        self.video.write(scaled)
                    count += 1
                break

            self.fpsClock.tick(25)

            if not inference_thread.is_alive():
                inference_thread = threading.Thread(target=self.infer)
                inference_thread.start()

            count += 1

        pygame.quit()
        if self.video is not None:
            print("🎥 Released video")
            self.video.release()


@click.command()
@click.option('-c', '--ckpt_path', required=True)
@click.option('-s', '--save_path', required=False)
@click.option('-m', '--run_multiple', required=False)
@click.option('-r', '--random_seed', required=False)
def main(ckpt_path: str, save_path: str, run_multiple: str, random_seed: str = None):
    if not run_multiple:
        action_predictor(
            ckpt_path,
            {'pos': None, 'sheep_pos': None, 'sheep_com': None, 'goal': None},
            'cpu',
            seed=None,
            save_path=save_path,
            param_dict={'num_sheep': 5, 'num_dog': 2},
            model_num_sheep=5).run()
    else:
        count = 1
        while count < int(run_multiple)+1:
            seed = None if random_seed else count
            action_predictor(
                ckpt_path,
                {'pos': None, 'sheep_pos': None, 'sheep_com': None, 'goal': None},
                'cpu',
                seed=seed,
                save_path=save_path+f'{count}',
                param_dict={'num_sheep': 5, 'num_dog': 2},
                model_num_sheep=5).run()
            count += 1


if __name__ == '__main__':
    main()