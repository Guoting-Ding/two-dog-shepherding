#!/usr/bin/env python
import csv
import threading
import time
from collections import deque
from enum import Enum
from enum import auto as enum_auto
from typing import Dict

import dill
import hydra
import numpy as np
import pygame
import torch
import yaml
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from shepherd_game.game import Game


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

        # Convert to torch Tensors of the right shape
        obs_data_tensors = dict_apply(
            {"image": images, "pos": pos},
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
            # 'image': np.transpose(pygame.surfarray.array3d(
            #     self.env.screen.convert()), (2, 0, 1)).astype(np.float32),
            'image': np.transpose(pygame.surfarray.array3d(
                self.env.screen.convert()), (2, 1, 0)).astype(np.float32),
            'pos': self.env.dog.astype(np.float32)
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


if __name__ == '__main__':
    action_predictor(
        '../nb/latest.ckpt', 'cuda:0').run()
