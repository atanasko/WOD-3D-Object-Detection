import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from PIL import Image

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
