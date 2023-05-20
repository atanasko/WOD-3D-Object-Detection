import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
