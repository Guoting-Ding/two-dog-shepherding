import os
import sys
from typing import List, Tuple

import click
import cv2
import numpy as np
from omegaconf import OmegaConf

from diffusion_policy.common.replay_buffer import ReplayBuffer


@click.command()
@click.option('-c', '--config', required=True)
@click.option('-r', '--rotate', required=True)
def main(config, rotate):
    """
    Convert the csv and bmp files from the shepherding game to zarr,
    the data format used for training models in this repo.

    Args:
        config (str): data_conversion config that defines how to convert the csv files
            into the input/output formats for the diffusion policy model.
        rotate (bool): rotate existing data to create more data
    """
    cfg = OmegaConf.load(config)

    # Build list of dirs to include
    dir_list = []
    for section in cfg.data:
        dir_list.extend(list(range(section['start'], section['end'] + 1)))

    # Init replay buffer for handling output
    zarr_path = str(cfg.output_file + '.zarr')
    replay_buffer = ReplayBuffer.create_from_path(
        zarr_path=zarr_path, mode='w')

    count = 0
    print_progress_bar(count, len(dir_list))
    # Add data to zarr, each directory contains one episode
    for dir in dir_list:
        path = cfg.data_dir + str(dir)

        # Read all images into an array
        img_files = sorted(os.listdir(path+"/img/"),
                           key=lambda x: int(os.path.splitext(x)[0]))
        img_list = []
        for img in img_files:
            img_list.append(cv2.imread(path+"/img/"+img))

        img_list = np.array(img_list)
        pos_list = np.loadtxt(path + "/pos.csv", delimiter=',', skiprows=0)
        sheep_pos = np.loadtxt(path + "/sheep_pos.csv",
                               delimiter=',', skiprows=0)
        episode = {
            "img": img_list,
            "pos": pos_list[:, 0:2],
            "action": pos_list[:, 2:4],
            "com": pos_list[:, 4:6],
            "dist": pos_list[:, 6:7],
            "sheep_pos": sheep_pos,
            "goal": pos_list[:, 7:9],
        }
        replay_buffer.add_episode(episode, compressors='disk')

        if rotate.lower() == 'true':
            add_rotations(img_list=img_list, pos_list=pos_list,
                          sheep_pos=sheep_pos, replay_buff=replay_buffer)

        count += 1
        print_progress_bar(count, len(dir_list))

    print(
        f"\nConverted {replay_buffer.n_episodes}",
        f"episodes to zarr format successfully and saved to {zarr_path}")


def add_rotations(img_list, pos_list, sheep_pos, replay_buff):
    for r in range(1, 4):
        rotated_images = [cv2.rotate(
            image, cv2.ROTATE_90_COUNTERCLOCKWISE) for image in img_list]
        rotated_pos = [rotate_pos(pos, r) for pos in pos_list[:, 0:2]]
        rotated_action = [rotate_pos(action, r) for action in pos_list[:, 2:4]]
        rotated_goal = [rotate_pos(goal, r) for goal in pos_list[:, 7:9]]

        rotated_sheep = np.copy(sheep_pos)

        # Loop through each row and rotate each pair (x, y) of coordinates
        # Iterate over each row (i-th point)
        for i in range(sheep_pos.shape[0]):
            # Iterate over each pair (x, y)
            for j in range(0, sheep_pos.shape[1], 2):
                rotated_x, rotated_y = rotate_pos(
                    [sheep_pos[i, j], sheep_pos[i, j + 1]],
                    r
                )
                # Store the rotated x and y values
                rotated_sheep[i, j] = rotated_x
                rotated_sheep[i, j + 1] = rotated_y

        rotated_com = [rotate_pos(com, r) for com in pos_list[:, 4:6]]
        episode = {
            "img": np.array(rotated_images),
            "pos": np.array(rotated_pos),
            "action": np.array(rotated_action),
            "com": np.array(rotated_com),
            "sheep_pos": np.array(rotated_sheep),
            "goal": np.array(rotated_goal),
        }
        replay_buff.add_episode(episode, compressors='disk')


def rotate_pos(pos: List[float], num: int) -> Tuple[float]:
    """
    Rotates the field coordinate by 90 degrees, ccw.

    Args:
        pos (List[float]): The x and y position
        num (int): Numer of 90 degree rotations to make

    Returns:
        List[float]: new x and y position
    """
    assert len(pos) == 2, "Postion must be x and y"
    if num <= 1:
        return [-pos[1]+150, pos[0]]
    else:
        return rotate_pos([-pos[1]+150, pos[0]], num-1)


def print_progress_bar(iteration: int, total: int, length: int = 40):
    """Print a progress bar to the terminal."""
    percent = (iteration / total)
    filled_length = int(length * percent)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r|{bar}| {percent:.1%} Complete')
    sys.stdout.flush()


if __name__ == '__main__':
    main()
