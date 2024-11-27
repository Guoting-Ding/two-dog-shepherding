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

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


class action_predictor:
    """Infer and predict actions for shepherding"""

    def __init__(self,
                 ckpt_path: str,
                 obs: Dict[str, any],
                 device: str = 'cuda:0',
                 seed: int = None,
                 save_path: str = None,
                 param_dict: Dict[str, any] = None,
                 label: str = None,
                 model_num_sheep: int = 1) -> None:
        """
        Create the action_predictor object.

        Args:
            ckpt_path (str): Path to the checkpoint file.
            obs (Dict[str, any]): Dictionary of observations to use.
            device (str): Device to use. Defaults to 'cuda:0'
            seed (Int | None): Seed to use for the game.
            save_path (str | None): Path to save the result.
            param_dict (Dict[str, any]): Parameter dict for the game environment.
            label (str | None): Text to label the video with. Appears in the top left
            model_num_sheep (int): Num of sheep the model is trained on

        """
        self.video: cv2.VideoWriter | None = None
        if save_path:
            if '.mp4' not in save_path:
                save_path = save_path+'.mp4'
            print("saving in: ", save_path)
            width = (FIELD_LENGTH+2*PADDING[0])*4
            height = (FIELD_LENGTH+2*PADDING[1])*4
            self.video = cv2.VideoWriter(
                save_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                15, (width, height))

        self.env = Game(
            seed=seed, **param_dict) if param_dict is not None else Game(seed=seed)
        self.fpsClock = pygame.time.Clock()

        # Load checkpoint
        self.payload = torch.load(
            open(ckpt_path, 'rb'), pickle_module=dill)
        self.cfg = self.payload['cfg']

        # Load workspace
        workspace_cls = hydra.utils.get_class(self.cfg._target_)
        self.workspace: BaseWorkspace = workspace_cls(self.cfg)
        self.workspace.load_payload(
            self.payload, exclude_keys=None, include_keys=None)

        # Load model
        self.policy: BaseImagePolicy = self.workspace.model
        if self.cfg.training.use_ema:
            self.policy = self.workspace.ema_model

        self.device = device
        self.policy.eval().to(self.device)

        # Create the observation queue
        self.obs_deque = deque(maxlen=self.policy.n_obs_steps)

        # Build sheep_pos update function based on expected # of agents
        if model_num_sheep == self.env.num_agents:
            def update_func(): return np.array(
                [pos for sheep in self.env.sheep for pos in sheep]).astype(np.float32)
        elif model_num_sheep > self.env.num_agents:
            def update_func():
                pos = np.array(
                    [pos for sheep in self.env.sheep for pos in sheep]).astype(np.float32)
                return np.tile(
                    pos, (model_num_sheep * 2 // (self.env.num_agents*2)) + 1)[:model_num_sheep * 2]
        elif model_num_sheep < self.env.num_agents:
            # 5 furthest sheep
            def update_func():
                sheep = self.env.sheep
                distances = np.array(
                    [dist(pos, self.env.target) for pos in sheep])
                sorted_dist = np.argsort(distances)[::-1]
                far_sheep = sheep[sorted_dist][:model_num_sheep]
                return np.array([pos for sheep in far_sheep for pos in sheep]).astype(np.float32)

        # Observations to use
        self.obs = dict.fromkeys(obs.keys())
        self.obs_updater = {
            'image': lambda: np.transpose(pygame.surfarray.array3d(self.env.screen.convert()), (2, 1, 0)).astype(np.float32),
            'pos': lambda: self.env.dog[0].astype(np.float32),
            'sheep_pos': update_func,
            'goal': lambda: self.env.target.astype(np.float32),
        }

        self.action = []
        self.label = label

    def infer(self) -> None:
        """Infer next set of actions"""
        # Stack observations from queue and convert to torch tensors
        obs_data_tensors = dict_apply({
            key: np.stack([data[key] for data in self.obs_deque])
            for key in self.obs_deque[0].keys()},
            lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))

        # Perform inference
        with torch.no_grad():
            result = self.policy.predict_action(obs_data_tensors)

        if len(self.action) == 0:
            self.action = result['action'].squeeze(
                0).to('cpu').numpy().tolist()

    def get_obs(self) -> Dict[str, np.ndarray]:
        """Get the current observations."""
        for key, update_func in self.obs_updater.items():
            if key in self.obs:
                self.obs[key] = update_func()

        # if image is not an observation, stack everything and return as 'obs'
        return self.obs if 'image' in self.obs else {
            'obs': np.hstack([self.obs[key] for key in self.obs.keys()])
        }

    def run(self, iters: int = 60*15) -> None:
        """
        Runs the shepherding game using inferred actions.

        Args:
            iters (int): Number of iterations to run
        """
        # Get initial obs and start inference
        obs = self.get_obs()
        self.obs_deque.append(obs)
        self.obs_deque.append(obs)
        inference_thread = threading.Thread(target=self.infer)
        inference_thread.start()

        count = 0
        while count < iters:
            self.env.render()

            # Execute the action
            done = False
            if len(self.action) > 0:
                action = self.action.pop()
                done = self.env.step(np.array([action]))
            else:
                # print("no action")
                continue

            # save observations
            obs = self.get_obs()
            self.obs_deque.append(obs)

            if self.video is not None:
                frame = np.transpose(pygame.surfarray.array3d(
                    self.env.screen), (1, 0, 2))

                # convert from rgb to bgr
                frame = frame[..., ::-1]

                # Scale up by a factor of 4 for a sharper image when scaled up
                scaled = np.repeat(np.repeat(frame, 4, axis=0), 4, axis=1)
                scaled = np.clip(scaled, 0, 255).astype(np.uint8)
                if self.label is not None:
                    cv2.putText(scaled, self.label, (15, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, BLACK, 2)
                self.video.write(scaled)

            if done:
                print("success")
                while count < iters:
                    # Fill up the video to 1000 frames
                    self.video.write(scaled)
                    count += 1
                print(f"filled up to {count} frames successfully")
                break

            # Update the game clock
            self.fpsClock.tick(15)

            if not inference_thread.is_alive():
                inference_thread = threading.Thread(target=self.infer)
                inference_thread.start()

            count += 1

        pygame.quit()
        if self.video is not None:
            print("Released video")
            self.video.release()


@ click.command()
@ click.option('-c', '--ckpt_path', required=True)
@ click.option('-s', '--save_path', required=False)
@ click.option('-m', '--run_multiple', required=False)
@ click.option('-r', '--random_seed', required=False)
def main(ckpt_path: str, save_path: str, run_multiple: str, random_seed: str = None):
    if not run_multiple:
        action_predictor(
            ckpt_path, 'cuda:0', seed=0, save_path=save_path).run()
    else:
        count = 1

        while count < int(run_multiple)+1:
            # use either 1 or random seed
            seed = None if random_seed else count

            action_predictor(
                ckpt_path, 'cuda:0',
                seed=seed, save_path=save_path+f'{count}').run()

            count += 1


if __name__ == '__main__':
    main()
