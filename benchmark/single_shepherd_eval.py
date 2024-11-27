import contextlib
import os
import sys
import tempfile

import click
import cv2
import numpy as np
from benchmark.action_predictor import action_predictor
from omegaconf import OmegaConf

from shepherd_game.parameters import FIELD_LENGTH, PADDING

# Lookup table
SHAPE = {
    1: (1, 1),
    2: (1, 2),
    3: (1, 3),
    4: (2, 2),
    6: (2, 3),
    8: (2, 4),
    9: (3, 3)
}

S_COLOR = np.array((255, 25, 25), dtype=np.uint8)


@ click.command()
@ click.option('-c', '--config', required=False)
def main(config: str):
    # Add file extension if not there and load config
    if '.yaml' not in config:
        config = config + '.yaml'

    cfg = OmegaConf.load(config)

    tf_list = []
    for each in cfg.runs:
        # Determine seed array
        seed_list = each['seed']
        if type(seed_list) == str:
            # assume random if the value is a string
            seed_list = [None for _ in range(0, each['num_of_runs'])]
        elif type(seed_list) == int:
            seed_list = [seed_list for _ in range(0, each['num_of_runs'])]

        # Use global config if local doesn't exist, or use None
        ckpt = each['ckpt_path'] if 'ckpt_path' in each else cfg.ckpt_path
        game_params = each['game_params'] if 'game_params' in each else None
        label = each['label'] if 'label' in each else None
        num_sheep = each['model_num_sheep'] if 'model_num_sheep' in each else cfg.model_num_sheep

        # Iterate through each seed and run
        for seed in seed_list:
            # Create a temp file to write the video to
            tf = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
            tf_list.append(tf)

            # Evaluate and save to the temp file, but with no output
            print(f"Evaluating {ckpt} with seed {seed}")
            # with open(os.devnull, 'w') as fnull:
            #     with contextlib.redirect_stderr(fnull), contextlib.redirect_stdout(fnull):
            action_predictor(ckpt_path=ckpt,
                             obs=each['obs'],
                             device=cfg.device,
                             seed=seed,
                             save_path=tf,
                             param_dict=game_params,
                             label=label,
                             model_num_sheep=num_sheep).run()

    # Create capture devices
    caps = [cv2.VideoCapture(video) for video in tf_list]

    # Create video writer
    # Multiply by 4 since action_predictor scales video up by 4
    frame_width = (FIELD_LENGTH+2*PADDING[0])*4
    frame_height = (FIELD_LENGTH+2*PADDING[1])*4
    y_size, x_size = SHAPE[len(tf_list)]
    shape = (
        frame_width*x_size + cfg.border*(x_size - 1),
        frame_height*y_size + cfg.border*(y_size - 1)
    )
    out = cv2.VideoWriter(
        cfg.output_path,                    # Filepath
        cv2.VideoWriter_fourcc(*'mp4v'),    # Video encoder
        int(15*cfg.playback_speed),         # fps
        shape                               # Pixel shape
    )

    # Build black lines to insert between frames
    black_line_vert = np.zeros(
        (frame_height, cfg.border, 3), dtype=np.uint8)
    black_line_horz = np.zeros(
        (cfg.border, shape[0], 3), dtype=np.uint8
    )

    # Path drawing params
    if cfg.draw_path:
        color_max = np.array(cfg.color_max, dtype=np.uint8)
        color_min = np.array(cfg.color_min, dtype=np.uint8)

    # Read from temp files and write to output video
    print("Beginning Video Stitching")
    shepherd_pos = []
    total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
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
        if all(not cap.isOpened() or cap.get(cv2.CAP_PROP_POS_FRAMES) >=
               cap.get(cv2.CAP_PROP_FRAME_COUNT) for cap in caps):
            break

        # Stitch together rows of images with a black bar in betwee
        row_list = []
        for i in range(y_size):
            # Split frames up into rows of the right size
            row = frames[i*x_size:(i+1)*x_size]

            # Make new list with an inserted black line between each frame
            row_concat = []
            for each in row:
                # Using numpy slicing causes issues since the
                # image and black line have different shapes
                row_concat.append(each)
                row_concat.append(black_line_vert)
            # Remove the extra line at the end
            row_concat.pop(-1)

            # Turn row into a single image
            row_list.append(np.hstack(row_concat))

        # Combine rows into a single image
        result = []
        for row in row_list:
            result.append(row)
            result.append(black_line_horz)
        # Remove the extra line at the end
        result.pop(-1)
        result = np.vstack(result)

        # Draw the path
        if cfg.draw_path:
            # Find contours that match the shepherd's color
            mask = cv2.inRange(result, color_min, color_max)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Loop through each contour to find their centers
            for contour in contours:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    shepherd_pos.append((cx, cy))

            # Draw all previous points
            for point in shepherd_pos:
                cv2.circle(result, point, cfg.path_size, cfg.path_color, -1)

        out.write(result)

        # Update pgrogress bar
        print_progress_bar(count, total_frames)
        count += 1

    # Send a new line to std out
    print_progress_bar(total_frames, total_frames)
    print()

    # Release capture devices
    for cap in caps:
        cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Clean up temp files
    for tf in tf_list:
        os.remove(tf)


def print_progress_bar(iteration: int, total: int, length: int = 40):
    """Print a progress bar to the terminal."""
    percent = (iteration / total)
    filled_length = int(length * percent)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r|{bar}| {percent:.1%} Complete')
    sys.stdout.flush()


if __name__ == "__main__":
    main()
