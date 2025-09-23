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

# line-buffered output
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

OmegaConf.register_new_resolver("eval", eval, replace=True)


class DualDogRunner:
    def __init__(self,
                 ckpt_path1: str,
                 ckpt_path2: str,
                 param_dict: Dict = None,
                 model_num_sheep: int = 5,
                 device: str = 'cpu'):

        if param_dict is None:
            param_dict = {}
        param_dict.setdefault('num_sheep', 5)
        param_dict.setdefault('num_dog', 2)

        param_dict.setdefault('sheep_top_right', True)




        self.env = Game(seed=None, **param_dict)
        self.fpsClock = pygame.time.Clock()
        self.device = device

        param_dict.setdefault('sheep_top_right', True)

        # === predictor 1 ===
        self.payload1 = torch.load(open(ckpt_path1, 'rb'), pickle_module=dill)
        self.cfg1 = self.payload1['cfg']
        self.workspace1 = hydra.utils.get_class(self.cfg1._target_)(self.cfg1)
        self.workspace1.load_payload(self.payload1)
        self.policy1: BaseImagePolicy = self.workspace1.ema_model if self.cfg1.training.use_ema else self.workspace1.model
        self.policy1.eval().to(device)
        self.obs_deque1 = deque(maxlen=self.policy1.n_obs_steps)
        self.action1 = []

        # === predictor 2 ===
        self.payload2 = torch.load(open(ckpt_path2, 'rb'), pickle_module=dill)
        self.cfg2 = self.payload2['cfg']
        self.workspace2 = hydra.utils.get_class(self.cfg2._target_)(self.cfg2)
        self.workspace2.load_payload(self.payload2)
        self.policy2: BaseImagePolicy = self.workspace2.ema_model if self.cfg2.training.use_ema else self.workspace2.model
        self.policy2.eval().to(device)
        self.obs_deque2 = deque(maxlen=self.policy2.n_obs_steps)
        self.action2 = []

        # === sheep pos updater ===
        if model_num_sheep == self.env.num_agents:
            self.get_sheep_pos = lambda: np.array([pos for sheep in self.env.sheep for pos in sheep]).astype(np.float32)
        elif model_num_sheep > self.env.num_agents:
            def tile_func():
                pos = np.array([pos for sheep in self.env.sheep for pos in sheep]).astype(np.float32)
                return np.tile(pos, (model_num_sheep * 2 // (self.env.num_agents * 2)) + 1)[:model_num_sheep * 2]
            self.get_sheep_pos = tile_func
        else:
            def farthest_func():
                sheep = self.env.sheep
                distances = np.array([dist(pos, self.env.target) for pos in sheep])
                sorted_dist = np.argsort(distances)[::-1]
                far_sheep = sheep[sorted_dist][:model_num_sheep]
                return np.array([pos for sheep in far_sheep for pos in sheep]).astype(np.float32)
            self.get_sheep_pos = farthest_func

    def get_obs1(self) -> Dict[str, np.ndarray]:
        return {
            'obs': np.hstack([
                self.env.dog[0].astype(np.float32),
                self.get_sheep_pos(),
                np.mean(np.array([pos for sheep in self.env.sheep for pos in sheep]).reshape(-1, 2), axis=0).astype(np.float32),
                self.env.target.astype(np.float32)
            ])
        }

    def get_obs2(self) -> Dict[str, np.ndarray]:
        return {
            'obs': np.hstack([
                self.env.dog[1].astype(np.float32),
                self.get_sheep_pos(),
                np.mean(np.array([pos for sheep in self.env.sheep for pos in sheep]).reshape(-1, 2), axis=0).astype(np.float32),
                self.env.target.astype(np.float32)
            ])
        }

    def infer1(self):
        obs_tensors = dict_apply({
            key: np.stack([data[key] for data in self.obs_deque1])
            for key in self.obs_deque1[0].keys()
        }, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))

        with torch.no_grad():
            result = self.policy1.predict_action(obs_tensors)
        self.action1 = result['action'].squeeze(0).to('cpu').numpy().tolist()

    def infer2(self):
        obs_tensors = dict_apply({
            key: np.stack([data[key] for data in self.obs_deque2])
            for key in self.obs_deque2[0].keys()
        }, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))

        with torch.no_grad():
            result = self.policy2.predict_action(obs_tensors)
        self.action2 = result['action'].squeeze(0).to('cpu').numpy().tolist()

    def run(self, iters: int = 60 * 15):
        self.obs_deque1.append(self.get_obs1())
        self.obs_deque1.append(self.get_obs1())
        self.obs_deque2.append(self.get_obs2())
        self.obs_deque2.append(self.get_obs2())

        thread1 = threading.Thread(target=self.infer1)
        thread2 = threading.Thread(target=self.infer2)
        thread1.start()
        thread2.start()

        count = 0
        while count < iters:
            self.env.render()

            if len(self.action1) > 0 and len(self.action2) > 0:
                a1 = self.action1.pop()
                a2 = self.action2.pop()
                done = self.env.step(np.array([a1, a2]))

                self.obs_deque1.append(self.get_obs1())
                self.obs_deque2.append(self.get_obs2())

                if done:
                    print("success")
                    break

                if not thread1.is_alive():
                    thread1 = threading.Thread(target=self.infer1)
                    thread1.start()
                if not thread2.is_alive():
                    thread2 = threading.Thread(target=self.infer2)
                    thread2.start()

                count += 1
                self.fpsClock.tick(15)
            else:
                continue

        pygame.quit()


@click.command()
@click.option('--ckpt_path1', required=True)
@click.option('--ckpt_path2', required=True)
def main(ckpt_path1: str, ckpt_path2: str):
    runner = DualDogRunner(ckpt_path1=ckpt_path1, ckpt_path2=ckpt_path2)
    runner.run()


if __name__ == '__main__':
    main()
