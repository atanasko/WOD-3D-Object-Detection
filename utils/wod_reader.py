import tensorflow as tf
import dask.dataframe as dd
from waymo_open_dataset import v2


def read_df(dataset_dir: str, context_name: str, tag: str) -> dd.DataFrame:
    """Creates a Dask DataFrame for the component specified by its tag."""
    paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/training_{tag}_{context_name}.parquet')
    return dd.read_parquet(paths)


def read_cam_img_cam_box_df(dataset_dir: str, context_name: str, camera_name: int):
    cam_img_df = read_df(dataset_dir, context_name, 'camera_image')
    cam_box_df = read_df(dataset_dir, context_name, 'camera_box')

    # Join all DataFrames using matching columns
    cam_img_df = cam_img_df[cam_img_df['key.camera_name'] == camera_name]
    cam_img_cam_box_df = v2.merge(cam_img_df, cam_box_df, right_group=True)

    return cam_img_cam_box_df
