#!/usr/bin/env python
import sys
import threading
from collections import deque
from typing import Dict

import click
import dill
import hydra
import numpy as np
import pygame
import torch
from hydra import compose
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from shepherd_game.game import Game
from shepherd_game.utils import dist

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


class action_predictor:
    """Infer and predict actions for shepherding"""

    def __init__(self, ckpt_path: str, device: str) -> None:
        """
        Create the action_predictor object.

        Args:
            ckpt_path (str): Path to the checkpoint file.
        """
        self.env = Game()
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

        # Flags
        self.running_inference = False

        # Action prediction latency
        self.latency_counter = 0

        # Predicted actions
        self.action = []

        self.inference_thread = threading.Thread()

    def infer(self) -> None:
        """Infer next set of actions"""
        self.running_inference = True

        # stack the last obs_horizon number of observations
        # shape = (2, 3, 130, 130)
        images = np.stack([x['image'] for x in self.obs_deque])

        # shape = (2, 2)
        pos = np.stack([x['pos'] for x in self.obs_deque])

        com = np.stack([x['com'] for x in self.obs_deque])
        dist = np.stack([x['dist'] for x in self.obs_deque])

        # Convert to torch Tensors of the right shape
        obs_data_tensors = dict_apply({
            "image": images,
            "pos": pos,
            "com": com,
            "dist": dist,
        },
            lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)
        )

        # Perform inference
        with torch.no_grad():
            result = self.policy.predict_action(obs_data_tensors)

        while len(self.action) > 0:
            pass
        self.action = result['action'].squeeze(
            0).to('cpu').numpy().tolist()

        self.running_inference = False

    def get_obs(self) -> None:
        return {
            'image': np.transpose(pygame.surfarray.array3d(
                self.env.screen.convert()), (2, 1, 0)).astype(np.float32),
            'pos': self.env.dog.astype(np.float32),
            'com': self.env.CoM.astype(np.float32),
            'dist': dist(self.env.CoM, self.env.target).astype(np.float32),
        }

    def run(self) -> None:
        """
        Runs the shepherding game using inferred actions.
        """
        while True:
            self.env.render()

            # Execute the action
            if len(self.action) > 0:
                action = self.action.pop()
                self.env.step(np.array(action))

            else:
                self.env.step(np.array([0.0, 0.0]))

            # save observations
            self.obs_deque.append(self.get_obs())

            # Update the game clock
            self.fpsClock.tick(15)

            if not self.inference_thread.is_alive():
                self.inference_thread = threading.Thread(target=self.infer)
                self.inference_thread.start()


@click.command()
@click.option('-c', '--ckpt_path', required=True)
def main(ckpt_path: str):
    action_predictor(
        ckpt_path, 'cuda:0').run()


if __name__ == '__main__':
    main()
