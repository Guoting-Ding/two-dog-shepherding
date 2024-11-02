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
from omegaconf import OmegaConf
from pygame.locals import *

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from shepherd_game.game import BLACK, WHITE, Game
from shepherd_game.parameters import FIELD_LENGTH, PADDING, TARGET_RADIUS

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


class action_predictor:
    """Infer and predict actions for shepherding"""

    def __init__(self,
                 ckpt_path: str,
                 device: str = 'cuda:0',
                 seed: int = None,
                 save_path: str = None,
                 draw_path: bool = True,
                 param_dict: Dict[str, any] = None) -> None:
        """
        Create the action_predictor object.

        Args:
            ckpt_path (str): Path to the checkpoint file.
            device (str): Device to use. Defaults to 'cuda:0'
            seed (Int | None): Seed to use for the game.
            save_path (str | None): Path to save the result.
            draw_path (bool): Draw the previous path of the shepherd. Defaults
                to True.
            param_dict (Dict[str, any]): Parameter dict for the game environment.

        """
        self.img = True if "img" in ckpt_path else False
        self.video: cv2.VideoWriter | None = None
        if save_path:
            import cv2
            if '.mp4' not in save_path:
                save_path = save_path+'.mp4'
            print("saving in: ", save_path)
            width = (FIELD_LENGTH+2*PADDING[0])*4
            height = (FIELD_LENGTH+2*PADDING[1])*4
            self.video = cv2.VideoWriter(
                save_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                30.0, (width, height))

        self.env = Game(
            seed=seed, **param_dict) if param_dict is not None else Game(seed=seed)
        self.fpsClock = pygame.time.Clock()

        if draw_path:
            self.screen =\
                pygame.display.set_mode((FIELD_LENGTH+2*PADDING[0],
                                         FIELD_LENGTH+2*PADDING[1]),
                                        SCALED)

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
        self.draw_path = draw_path

        # Action prediction latency
        self.latency_counter = 0

        # Predicted actions
        self.action = []

        self.inference_thread = threading.Thread()

    def infer(self) -> None:
        """Infer next set of actions"""
        # stack the last obs_horizon number of observations
        # shape = (2, 3, 130, 130)
        if self.img:
            images = np.stack([x['image'] for x in self.obs_deque])

            # shape = (2, 2)
            pos = np.stack([x['pos'] for x in self.obs_deque])

            sheep_pos = np.stack([x['sheep_pos'] for x in self.obs_deque])

            # com = np.stack([x['com'] for x in self.obs_deque])
            # dist = np.stack([x['dist'] for x in self.obs_deque])

            # Convert to torch Tensors of the right shape
            obs_data_tensors = dict_apply({
                "image": images,
                "pos": pos,
                "sheep_pos": sheep_pos
                # "com": com,
                # "dist": dist,
            },
                lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)
            )
        else:
            obs = np.stack([x['obs'] for x in self.obs_deque])
            obs_data_tensors = {
                'obs': torch.from_numpy(
                    obs).unsqueeze(0).to(self.device)}

        # Perform inference
        with torch.no_grad():
            result = self.policy.predict_action(obs_data_tensors)

        if len(self.action) == 0:
            self.action = result['action'].squeeze(
                0).to('cpu').numpy().tolist()

    def get_obs(self) -> None:
        return {
            'image': np.transpose(pygame.surfarray.array3d(
                self.env.screen.convert()), (2, 1, 0)).astype(np.float32),
            'pos': self.env.dog.astype(np.float32),
            'sheep_pos': np.array([pos for sheep in self.env.sheep for pos in sheep]).astype(np.float32)
            # 'com': self.env.CoM.astype(np.float32),
            # 'dist': dist(self.env.CoM, self.env.target).astype(np.float32),
        } if self.img else {
            'obs': np.hstack([
                self.env.dog.astype(np.float32),
                np.array([pos for sheep in self.env.sheep for pos in sheep]).astype(
                    np.float32)
            ])
        }

    def run(self, iters: int = 60*15) -> None:
        """
        Runs the shepherding game using inferred actions.

        Args:
            iters (int): Number of iterations to run
        """
        count = 0
        while count < iters:
            self.env.render(draw=(not self.draw_path))

            # Execute the action
            done = False
            if len(self.action) > 0:
                action = self.action.pop()
                done = self.env.step(np.array(action))

            else:
                done = self.env.step(np.array([0.0, 0.0]))

            # save observations
            obs = self.get_obs()
            self.obs_deque.append(obs)

            # After saving the observation, we can draw in the new path
            # This prevents the path from interfering with observations and inferences
            if self.draw_path:
                self.screen.fill((19, 133, 16))

                # Draw all previous points
                for data in self.env.pos:
                    x, y = data[0], data[1]
                    pygame.draw.circle(self.screen,
                                       (50, 242, 255),
                                       (x + PADDING[0], y+PADDING[0]),
                                       1, 0)

                # Draw game elements, detailed in shepherd_game/game.py:Game().render()
                pygame.draw.circle(self.screen, BLACK, tuple(
                    PADDING + self.env.target), TARGET_RADIUS, 0)
                pygame.draw.circle(self.screen, (25, 25, 255), tuple(
                    PADDING + self.env.dog), 1, 0)
                [pygame.draw.circle(self.screen, WHITE, tuple(PADDING + a), 1, 0)
                    for a in self.env.sheep]
                pygame.draw.circle(self.screen, BLACK,
                                   tuple(PADDING + self.env.CoM), 2, 0)
                pygame.draw.circle(self.screen, (200, 200, 200),
                                   tuple(PADDING + self.env.CoM), 1, 0)
                pygame.display.update()

            if self.video is not None:
                screen = self.screen if self.draw_path else self.env.screen
                frame = np.transpose(pygame.surfarray.array3d(
                    screen), (1, 0, 2))

                # convert from rgb to bgr
                frame = frame[..., ::-1]

                # Scale up by a factor of 4 for a sharper image when scaled up
                scaled = np.repeat(np.repeat(frame, 4, axis=0), 4, axis=1)
                scaled = np.clip(scaled, 0, 255).astype(np.uint8)
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

            if not self.inference_thread.is_alive():
                self.inference_thread = threading.Thread(target=self.infer)
                self.inference_thread.start()

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
                seed=0, save_path=save_path+f'{count}').run()

            count += 1


if __name__ == '__main__':
    main()
