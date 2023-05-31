import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import open3d as o3d
from PIL import Image
import utils.config as config

colormap = {1: "red", 2: "blue", 3: "yellow", 4: "yellow"}


def vis_cam_img(camera_image, camera_box):
    image = Image.open(io.BytesIO(camera_image.image))

    ax = plt.subplot()
    # Iterate over the individual labels.
    for j, (object_id, object_type, x, size_x, y, size_y) in enumerate(zip(
            camera_box.key.camera_object_id, camera_box.type, camera_box.box.center.x, camera_box.box.size.x,
            camera_box.box.center.y,
            camera_box.box.size.y
    )):
        # Draw the object bounding box.
        ax.add_patch(patches.Rectangle(
            xy=(x - 0.5 * size_x, y - 0.5 * size_y),
            width=size_x,
            height=size_y,
            linewidth=1,
            edgecolor=colormap[object_type],
            facecolor='none'))

    plt.imshow(image)
    plt.show()


def vis_range_image(range_image):
    # get range image numpy array
    ri = range_image.tensor.numpy()

    # set negative range values to 0
    ri[ri < 0] = 0.0

    # extract range and intensity channels from range image
    ri_range = ri[:, :, 0]
    ri_intensity = ri[:, :, 1]

    # map range to 255 values
    ri_range = ri_range * 255 / (np.amax(ri_range) - np.amin(ri_range))

    # map intensity to 255 values, and normalize to 99% - 1% max intensity to mitigate influence of outliers
    ri_intensity = np.amax(ri_intensity) * (0.99 - 0.01) * ri_intensity * 255 / (
            np.amax(ri_intensity) - np.amin(ri_intensity))

    ri_range_intensity = np.vstack((ri_range, ri_intensity))
    ri_range_intensity = ri_range_intensity.astype(np.uint8)

    # plt.imshow(img_range_intensity)
    # plt.show()

    cv2.imshow("Range image", ri_range_intensity)
    cv2.waitKey(0)


def vis_pcl(points):
    # pylint: disable=no-member (E1101)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    pcd = o3d.geometry.PointCloud()

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.6, origin=[0, 0, 0])

    pcd.points = o3d.utility.Vector3dVector(points.numpy())

    vis.add_geometry(pcd)
    vis.add_geometry(mesh_frame)

    vis.run()


def vis_bev_image(bev_img, lidar_box):
    cfg = config.load()

    bev_img = cv2.rotate(bev_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # bev_img = cv2.rotate(bev_img, cv2.ROTATE_90_CLOCKWISE)
    # bev_img = cv2.rotate(bev_img, cv2.ROTATE_180)
    img = Image.fromarray(bev_img).convert('RGB')

    discrete = (cfg.range_x[1] - cfg.range_x[0]) / cfg.bev_width

    ax = plt.subplot()
    # Iterate over the individual labels.
    for j, (object_id, object_type, x, size_x, y, size_y, yaw) in enumerate(zip(
            lidar_box.key.laser_object_id, lidar_box.type, lidar_box.box.center.x, lidar_box.box.size.x,
            lidar_box.box.center.y,
            lidar_box.box.size.y, lidar_box.box.heading
    )):
        # Draw the object bounding box.
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        x = x / discrete
        y = (-y / discrete) + cfg.bev_width / 2

        size_x = size_x / discrete
        size_y = size_y / discrete

        size_x = size_x * cos_yaw + size_y * sin_yaw
        size_y = - size_x * sin_yaw + size_y * cos_yaw

        ax.add_patch(patches.Rectangle(
            xy=(x - 0.5 * size_x,
                y - 0.5 * size_y),
            width=size_x,
            height=size_y,
            linewidth=1,
            edgecolor=colormap[object_type],
            facecolor='none'))

    plt.imshow(img)
    plt.show()
