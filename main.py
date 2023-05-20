import sys
import argparse
import utils.wod_reader as wod_reader
import utils.visualizer as vis
from waymo_open_dataset import v2


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir')
    parser.add_argument('-c', '--context_name')
    args = parser.parse_args()

    if args.dataset_dir is None or args.context_name is None:
        parser.print_help()
        sys.exit()

    return args.dataset_dir, args.context_name


def vis_cam_img(dataset_dir: str, context_name: str, camera_name: int):
    cam_img_cam_box_df = wod_reader.read_cam_img_cam_box_df(dataset_dir, context_name, camera_name)

    for i, (_, r) in enumerate(cam_img_cam_box_df.iterrows()):
        cam_image = v2.CameraImageComponent.from_dict(r)
        cam_box = v2.CameraBoxComponent.from_dict(r)

        vis.vis_cam_img(cam_image, cam_box)


def main():
    dataset_dir, context_name = parse_arguments()

    vis_cam_img(dataset_dir, context_name, 1)


# object detector entry point
if __name__ == '__main__':
    main()
