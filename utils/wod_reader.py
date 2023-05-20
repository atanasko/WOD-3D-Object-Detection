import tensorflow as tf
import dask.dataframe as dd


def read_dataframe(dataset_dir: str, tag: str, context_name: str) -> dd.DataFrame:
    """Creates a Dask DataFrame for the component specified by its tag."""
    paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/training_{tag}_{context_name}.parquet')
    return dd.read_parquet(paths)
