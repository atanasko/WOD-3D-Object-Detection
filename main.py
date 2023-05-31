import sys
import argparse
import numpy as np
import utils.wod_reader as wod_reader
import utils.visualizer as vis
from waymo_open_dataset import v2
# from waymo_open_dataset.v2.perception.utils import lidar_utils as _lidar_utils
import utils.lidar_utils as _lidar_utils
import utils.config as config


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


def pcl_to_bev(pcl):
    cfg = config.load()

    pcl_npa = pcl.numpy()
    mask = np.where((pcl_npa[:, 0] >= cfg.range_x[0]) & (pcl_npa[:, 0] <= cfg.range_x[1]) &
                    (pcl_npa[:, 1] >= cfg.range_y[0]) & (pcl_npa[:, 1] <= cfg.range_y[1]) &
                    (pcl_npa[:, 2] >= cfg.range_z[0]) & (pcl_npa[:, 2] <= cfg.range_z[1]))
    pcl_npa = pcl_npa[mask]

    # compute bev-map discretization by dividing x-range by the bev-image height
    bev_discrete = (cfg.range_x[1] - cfg.range_x[0]) / cfg.bev_height

    # create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates
    pcl_cpy = np.copy(pcl_npa)
    pcl_cpy[:, 0] = np.int_(np.floor(pcl_cpy[:, 0] / bev_discrete))

    # transform all metrix y-coordinates as well but center the forward-facing x-axis in the middle of the image
    pcl_cpy[:, 1] = np.int_(np.floor(pcl_cpy[:, 1] / bev_discrete) + (cfg.bev_width + 1) / 2)

    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    pcl_cpy[:, 2] = pcl_cpy[:, 2] - cfg.range_z[0]

    # re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then by decreasing height
    idx_height = np.lexsort((-pcl_cpy[:, 2], pcl_cpy[:, 1], pcl_cpy[:, 0]))
    lidar_pcl_hei = pcl_cpy[idx_height]

    # extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    _, idx_height_unique = np.unique(lidar_pcl_hei[:, 0:2], axis=0, return_index=True)
    lidar_pcl_hei = lidar_pcl_hei[idx_height_unique]

    # assign the height value of each unique entry in lidar_top_pcl to the height map and
    # make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    height_map = np.zeros((cfg.bev_height + 1, cfg.bev_width + 1))
    height_map[np.int_(lidar_pcl_hei[:, 0]), np.int_(lidar_pcl_hei[:, 1])] = lidar_pcl_hei[:, 2] / float(
        np.abs(cfg.range_z[1] - cfg.range_z[0]))

    # sort points such that in case of identical BEV grid coordinates, the points in each grid cell are arranged based on their intensity
    pcl_cpy[pcl_cpy[:, 2] > 1.0, 2] = 1.0
    idx_intensity = np.lexsort((-pcl_cpy[:, 2], pcl_cpy[:, 1], pcl_cpy[:, 0]))
    pcl_cpy = pcl_cpy[idx_intensity]

    # only keep one point per grid cell
    _, indices = np.unique(pcl_cpy[:, 0:2], axis=0, return_index=True)
    lidar_pcl_int = pcl_cpy[indices]

    # create the intensity map
    intensity_map = np.zeros((cfg.bev_height + 1, cfg.bev_width + 1))
    intensity_map[np.int_(lidar_pcl_int[:, 0]), np.int_(lidar_pcl_int[:, 1])] = lidar_pcl_int[:, 2] / (
            np.amax(lidar_pcl_int[:, 2]) - np.amin(lidar_pcl_int[:, 2]))

    # Compute density layer of the BEV map
    density_map = np.zeros((cfg.bev_height + 1, cfg.bev_width + 1))
    _, _, counts = np.unique(lidar_pcl_int[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalized_counts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    density_map[np.int_(lidar_pcl_int[:, 0]), np.int_(lidar_pcl_int[:, 1])] = normalized_counts

    # visualize height map
    # img_height = height_map * 256
    # img_height = img_height.astype(np.uint8)

    # # visualize intensity map
    # img_intensity = intensity_map * 256
    # img_intensity = img_intensity.astype(np.uint8)

    bev_map = np.zeros((3, cfg.bev_height, cfg.bev_width))
    bev_map[2, :, :] = density_map[:cfg.bev_height, :cfg.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:cfg.bev_height, :cfg.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:cfg.bev_height, :cfg.bev_width]  # b_map

    bev_map = (np.transpose(bev_map, (1, 2, 0)) * 255).astype(np.uint8)

    # return img_intensity
    return bev_map


def vis_bev_image(dataset_dir: str, context_name: str, laser_name: int):
    lidar_lidar_box_df = wod_reader.read_lidar_lidar_box_df(dataset_dir, context_name, laser_name)
    lidar_calibration_df = wod_reader.read_lidar_calibration_df(dataset_dir, context_name, laser_name)
    lidar_pose_df = wod_reader.read_lidar_pose_df(dataset_dir, context_name, laser_name)
    vehicle_pose_df = wod_reader.read_vehicle_pose_df(dataset_dir, context_name)

    df = lidar_lidar_box_df.merge(lidar_calibration_df)
    df = v2.merge(df, lidar_pose_df)
    df = v2.merge(df, vehicle_pose_df)

    for i, (_, r) in enumerate(df.iterrows()):
        lidar = v2.LiDARComponent.from_dict(r)
        lidar_box = v2.LiDARBoxComponent.from_dict(r)
        lidar_calibration = v2.LiDARCalibrationComponent.from_dict(r)
        lidar_pose = v2.LiDARPoseComponent.from_dict(r)
        vehicle_pose = v2.VehiclePoseComponent.from_dict(r)

        pcl = _lidar_utils.convert_range_image_to_point_cloud(lidar.range_image_return1, lidar_calibration,
                                                              lidar_pose.range_image_return1, vehicle_pose)
        bev_img = pcl_to_bev(pcl)
        vis.vis_bev_image(bev_img, lidar_box)


def main():
    dataset_dir, context_name = parse_arguments()

    # vis_cam_img(dataset_dir, context_name, 1)
    # vis_range_image(dataset_dir, context_name, 1)
    # vis_pcl(dataset_dir, context_name, 1)
    vis_bev_image(dataset_dir, context_name, 1)


# object detector entry point
if __name__ == '__main__':
    main()
