import sys
import argparse
import utils.wod_reader as wod_reader
import utils.visualizer as vis
from waymo_open_dataset import v2
# from waymo_open_dataset.v2.perception.utils import lidar_utils as _lidar_utils
import utils.lidar_utils as _lidar_utils


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


def vis_range_image(dataset_dir: str, context_name: str, laser_name: int):
    lidar_df = wod_reader.read_lidar_df(dataset_dir, context_name, laser_name)

    for i, (_, r) in enumerate(lidar_df.iterrows()):
        lidar = v2.LiDARComponent.from_dict(r)

        vis.vis_range_image(lidar.range_image_return1)


def vis_pcl(dataset_dir: str, context_name: str, laser_name: int):
    lidar_df = wod_reader.read_lidar_df(dataset_dir, context_name, laser_name)
    lidar_calibration_df = wod_reader.read_lidar_calibration_df(dataset_dir, context_name, laser_name)
    lidar_pose_df = wod_reader.read_lidar_pose_df(dataset_dir, context_name, laser_name)
    vehicle_pose_df = wod_reader.read_vehicle_pose_df(dataset_dir, context_name)

    df = lidar_df.merge(lidar_calibration_df)
    df = v2.merge(df, lidar_pose_df)
    df = v2.merge(df, vehicle_pose_df)

    for i, (_, r) in enumerate(df.iterrows()):
        lidar = v2.LiDARComponent.from_dict(r)
        lidar_calibration = v2.LiDARCalibrationComponent.from_dict(r)
        lidar_pose = v2.LiDARPoseComponent.from_dict(r)
        vehicle_pose = v2.VehiclePoseComponent.from_dict(r)

        points = _lidar_utils.convert_range_image_to_point_cloud(lidar.range_image_return1, lidar_calibration,
                                                                 lidar_pose.range_image_return1, vehicle_pose)
        vis.vis_pcl(points)


def main():
    dataset_dir, context_name = parse_arguments()

    # vis_cam_img(dataset_dir, context_name, 1)
    # vis_range_image(dataset_dir, context_name, 1)
    vis_pcl(dataset_dir, context_name, 1)


# object detector entry point
if __name__ == '__main__':
    main()
