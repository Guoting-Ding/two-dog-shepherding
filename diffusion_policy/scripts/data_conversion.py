import os
from typing import List, Tuple

import click
import cv2
import numpy as np
from omegaconf import OmegaConf

from diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer


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
            "sheep_pos": sheep_pos
        }
        replay_buffer.add_episode(episode, compressors='disk')

        # TODO: Update this to take in position and action and sheep_pos
        if rotate.lower == 'true':
            # Rotate 90 degrees
            add_rotation(img_list=img_list, pos_list=pos_list,
                         replay_buff=replay_buffer)
            # Rotate 180 degrees
            add_rotation(img_list=img_list, pos_list=pos_list,
                         replay_buff=replay_buffer)
            # Rotate 270 degrees
            add_rotation(img_list=img_list, pos_list=pos_list,
                         replay_buff=replay_buffer)

    print(
        f"Converted {replay_buffer.n_episodes}",
        f"episodes to zarr format successfully and saved to {zarr_path}")


def add_rotation(img_list, pos_list, replay_buff):
    rotated_images = [cv2.rotate(
        image, cv2.ROTATE_90_COUNTERCLOCKWISE) for image in img_list]
    rotated_pos = [rotate_pos(pos) for pos in pos_list]

    episode = {
        "img": np.array(rotated_images),
        "action": np.array(rotated_pos)
    }
    replay_buff.add_episode(episode, compressors='disk')


def rotate_pos(pos: List[float]) -> Tuple[float]:
    """
    Rotates the field coordinate by 90 degrees, ccw.

    Args:
        pos (List[float]): The x and y position

    Returns:
        List[float]: new x and y position
    """
    assert len(pos) == 2, "Postion must be x and y"
    return [-pos[1]+150, pos[0]]


if __name__ == '__main__':
    main()
