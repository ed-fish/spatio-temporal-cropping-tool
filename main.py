import argparse
import confuse
import os
import logging
import glob
import pickle
import cv2
import torch
from transforms.img_transforms import ImgTransform, Normaliser
from models.models import ImgEmbeddingExtractor
import csv


def init_models(config):
    embedder = ImgEmbeddingExtractor(config)
    return embedder


def get_clips(video_path, config):
    print(video_path)
    if os.path.isdir(video_path):
        for file_path in os.listdir(video_path):
            file_path = os.path.join(video_path, file_path)
            print(file_path)
            get_transposed_crops(file_path, config)
    else:
        get_transposed_crops(video_path, config)


def get_transposed_crops(video_file, config):

    # Split the video into chunks of frames - n_frames defined in config.yaml
    chunk_list = split_frames(video_file, config["frame_length"].get())
    out_dir = config["output"].get()
    os.makedirs(out_dir, exist_ok=True)

    # Enumerate through the groups of images and save them to a directory
    for i, chunk, in enumerate(chunk_list):
        im_count = 0
        # Root directory for the chunk : /scene/chunkn/
        output_root_dir = os.path.join(
            out_dir,  video_file.split("/")[-1], str(i))
        os.makedirs(output_root_dir, exist_ok=True)
        img_dir = os.path.join(output_root_dir, "original_imgs")
        output_img_dir = os.path.join(output_root_dir, "cropped_images")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(output_img_dir, exist_ok=True)

        if config["tidy_start"].get():
            chunk = chunk[1:]
        video_stack = []

        for i, img in enumerate(chunk):

            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(img_dir,
                        str(i) + ".png"), img)

            # Transform and normalise the images.
            transformer = ImgTransform(img, video_file, config)
            transformed_img = transformer.transform_with_prob(img)
            cv2.imwrite(os.path.join(output_img_dir,
                        str(i) + ".png"), transformed_img)


def split_frames(video_file, chunk_length):
    frame_list = []
    chunk_list = []
    vidcap = cv2.VideoCapture(video_file)
    success, image = vidcap.read()
    frame_list.append(image)

    while success:
        success, image = vidcap.read()
        if success:
            frame_list.append(image)
            if len(frame_list) == chunk_length + 1:
                chunk_list.append(frame_list)
                frame_list = []

    return chunk_list


def main():

    # Setup passing and logger

    logging.basicConfig(level=logging.DEBUG)
    config = confuse.Configuration("Embedingator")
    config.set_file("config.yaml")
    parser = argparse.ArgumentParser()

    # Args

    parser.add_argument("--root", help="Specify the root data directory")
    parser.add_argument(
        "--config",
        default=os.path.join(os.getcwd(), "config.yaml"),
        help="Path to config.yaml file",
    )
    parser.add_argument(
        "--video", help="The location of the video file or directory - leave blank if using config for the input")
    parser.add_argument("--output", help="Set output directory")
    parser.add_argument("--transform_prob",
                        help="Set probability of transform from 0.0 to 1.0")
    parser.add_argument(
        "--frame_length", help="Set the number of frames for each crop", type=int)
    parser.add_argument("--gpu", help="Set the gpu number - 0 by default")

    # Get args and set config

    args = parser.parse_args()
    config.set_args(args)

    # If no arg parsed - default to current directory

    try:
        root = config["video"].get()
    except confuse.exceptions.NotFoundError:
        logging.info("No root directory set, using this dir")
        root = os.getcwd()

    get_clips(config['video'].get(), config)


if __name__ == "__main__":
    main()
