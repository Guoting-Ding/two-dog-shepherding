import os
import tempfile
from typing import Tuple

import click
import cv2
import dill
import hydra
import numpy as np
from omegaconf import OmegaConf

from diffusion_policy.benchmark.action_predictor import action_predictor
from shepherd_game.parameters import FIELD_LENGTH, PADDING

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

# Lookup table
SHAPE = {
    2: (1, 2),
    3: (1, 3),
    4: (2, 2),
    6: (2, 3),
    8: (2, 4),
    9: (3, 3)
}


@ click.command()
@ click.option('-c', '--config', required=False)
def main(config: str):
    # Add file extension if not there and load config
    if '.yaml' not in config:
        config = config + '.yaml'

    cfg = OmegaConf.load(config)

    for each in cfg.runs:
        assert each['num_of_runs'] in SHAPE, "invalid num of runs"

        # Determine seed array
        seed_list = each['seed']
        if type(seed_list) == str:
            # assume random if the value is a string
            seed_list = [None for _ in range(0, each['num_of_runs'])]
        elif type(seed_list) == int:
            seed_list = [seed_list for _ in range(0, each['num_of_runs'])]
        else:
            assert len(seed_list) == each['num_of_runs'], \
                "number of runs not equal to seed list"

        # Iterate through each seed and run
        tf_list = []
        for seed in seed_list:
            # Create a temp file to write the video to
            tf = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
            tf_list.append(tf)

            # Evaluate and save to the temp file
            action_predictor(ckpt_path=cfg.ckpt_path, device=cfg.device,
                             seed=seed, save_path=tf, draw_path=True).run()

        # Create capture devices
        caps = [cv2.VideoCapture(video) for video in tf_list]

        # Create video writer with shape
        x_size, y_size = SHAPE(len(tf_list))

        # Multiply by 4 since action_predictor scales up by 4
        frame_width = (FIELD_LENGTH+2*PADDING[0])*4
        frame_height = (FIELD_LENGTH+2*PADDING[1])*4
        shape = (
            frame_width*x_size + cfg.border*(x_size - 1),
            frame_height*y_size + cfg.border*(y_size - 1)
        )
        out = cv2.VideoWriter(
            each['output_path'],                # Filepath
            cv2.VideoWriter_fourcc(*'mp4v'),    # Video encoder
            int(15*each['speed']),              # fps
            shape                               # Pixel shape
        )

        # Read from temp files and write to output video
        while True:
            frames = []
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    frames.append(
                        np.zeros((frame_height, frame_width, 3), dtype=np.uint8))
                else:
                    frames.append(frame)

            # Break the loop if all videos have ended
            if all(not cap.isOpened() or cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(cv2.CAP_PROP_FRAME_COUNT) for cap in caps):
                break

            # Create the output frame by stacking based on the SHAPE
            # black_line = np.zeros(
            #     (frame_height, cfg.border, 3), dtype=np.uint8)

            # row1 = np.hstack(
            #     [frames[0], black_line, frames[1], black_line, frames[2]])
            # row2 = np.hstack(
            #     [frames[3], black_line, frames[4], black_line, frames[5]])

            # black_line = np.zeros(
            #     (2, (frame_width + 2) * cfg.border, 3), dtype=np.uint8)
            # output_frame = np.vstack([row1, black_line, row2])

            # Write the output frame to the video
            out.write(output_frame)

    # Clean up temp files
    for tf in tf_list:
        os.remove(tf)


if __name__ == "__main__":
    main()
