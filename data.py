import json
import os
import csv
import math
import random
import logging
import tempfile
import time
import shutil
import multiprocessing
from multiprocessing.dummy import Pool
from ast import literal_eval as make_tuple
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

import deepchem as dc
from deepchem.utils.typing import OneOrMany, Shape
from deepchem.utils.data_utils import save_to_disk, load_from_disk, load_image_files

Batch = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

logger = logging.getLogger(__name__)


def sparsify_features(X: np.ndarray) -> np.ndarray:
    n_samples = len(X)
    X_sparse = []
    for i in range(n_samples):
        nonzero_inds = np.nonzero(X[i])[0]
        nonzero_vals = X[i][nonzero_inds]
        X_sparse.append((nonzero_inds, nonzero_vals))
    return np.array(X_sparse, dtype=object)


def densify_features(X_sparse: np.ndarray, num_features: int) -> np.ndarray:
    n_samples = len(X_sparse)
    X = np.zeros((n_samples, num_features))
    for i in range(n_samples):
        nonzero_inds, nonzero_vals = X_sparse[i]
        X[i][nonzero_inds.astype(int)] = nonzero_vals
    return X


def pad_features(batch_size: int, X_b: np.ndarray) -> np.ndarray:
    num_samples = len(X_b)
    if num_samples > batch_size:
        raise ValueError("Cannot pad an array longer than `batch_size`")
    elif num_samples == batch_size:
        return X_b
    else:
        # By invariant of when this is called, can assume num_samples > 0
        # and num_samples < batch_size
        if len(X_b.shape) > 1:
            feature_shape = X_b.shape[1:]
            X_out = np.zeros((batch_size,) + feature_shape, dtype=X_b.dtype)
        else:
            X_out = np.zeros((batch_size,), dtype=X_b.dtype)

        # Fill in batch arrays
        start = 0
        while start < batch_size:
            num_left = batch_size - start
            if num_left < num_samples:
                increment = num_left
            else:
                increment = num_samples
            X_out[start:start + increment] = X_b[:increment]
            start += increment
        return X_out


def pad_batch(batch_size: int, X_b: np.ndarray, y_b: np.ndarray,
              w_b: np.ndarray, ids_b: np.ndarray) -> Batch:
    num_samples = len(X_b)
    if num_samples == batch_size:
        return (X_b, y_b, w_b, ids_b)
    # By invariant of when this is called, can assume num_samples > 0
    # and num_samples < batch_size
    if len(X_b.shape) > 1:
        feature_shape = X_b.shape[1:]
        X_out = np.zeros((batch_size,) + feature_shape, dtype=X_b.dtype)
    else:
        X_out = np.zeros((batch_size,), dtype=X_b.dtype)

    if y_b is None:
        y_out = None
    elif len(y_b.shape) < 2:
        y_out = np.zeros(batch_size, dtype=y_b.dtype)
    else:
        y_out = np.zeros((batch_size,) + y_b.shape[1:], dtype=y_b.dtype)

    if w_b is None:
        w_out = None
    elif len(w_b.shape) < 2:
        w_out = np.zeros(batch_size, dtype=w_b.dtype)
    else:
        w_out = np.zeros((batch_size,) + w_b.shape[1:], dtype=w_b.dtype)

    ids_out = np.zeros((batch_size,), dtype=ids_b.dtype)

    # Fill in batch arrays
    start = 0
    # Only the first set of copy will be counted in training loss
    if w_out is not None:
        w_out[start:start + num_samples] = w_b[:]

    while start < batch_size:
        num_left = batch_size - start
        if num_left < num_samples:
            increment = num_left
        else:
            increment = num_samples
        X_out[start:start + increment] = X_b[:increment]

        if y_out is not None:
            y_out[start:start + increment] = y_b[:increment]

        ids_out[start:start + increment] = ids_b[:increment]
        start += increment

    return (X_out, y_out, w_out, ids_out)


class Dataset(object):
    def __init__(self) -> None:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def get_shape(self) -> Tuple[Shape, Shape, Shape, Shape]:
        raise NotImplementedError()

    def get_task_names(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def X(self) -> np.ndarray:

        raise NotImplementedError()

    @property
    def y(self) -> np.ndarray:

        raise NotImplementedError()

    @property
    def ids(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def w(self) -> np.ndarray:
        raise NotImplementedError()

    def __repr__(self) -> str:
        """Convert self to REPL print representation."""
        threshold = dc.utils.get_print_threshold()
        task_str = np.array2string(np.array(self.get_task_names()),
                                   threshold=threshold)
        X_shape, y_shape, w_shape, _ = self.get_shape()
        if self.__len__() < dc.utils.get_max_print_size():
            id_str = np.array2string(self.ids, threshold=threshold)
            return "<%s X.shape: %s, y.shape: %s, w.shape: %s, ids: %s, task_names: %s>" % (
                self.__class__.__name__, str(X_shape), str(y_shape),
                str(w_shape), id_str, task_str)
        else:
            return "<%s X.shape: %s, y.shape: %s, w.shape: %s, task_names: %s>" % (
                self.__class__.__name__, str(X_shape), str(y_shape),
                str(w_shape), task_str)

    def __str__(self) -> str:
        """Convert self to str representation."""
        return self.__repr__()

    def iterbatches(self,
                    batch_size: Optional[int] = None,
                    epochs: int = 1,
                    deterministic: bool = False,
                    pad_batches: bool = False) -> Iterator[Batch]:
       
        raise NotImplementedError()

    def itersamples(self) -> Iterator[Batch]:
        raise NotImplementedError()

    def transform(self, transformer: "dc.trans.Transformer",
                  **args) -> "Dataset":
        raise NotImplementedError()

    def select(self,
               indices: Union[Sequence[int], np.ndarray],
               select_dir: Optional[str] = None) -> "Dataset":
        raise NotImplementedError()

    def get_statistics(self,
                       X_stats: bool = True,
                       y_stats: bool = True) -> Tuple[np.ndarray, ...]:
        x_shape, y_shape, w_shape, ids_shape = self.get_shape()
        X_means = np.zeros(x_shape[1:])
        X_m2 = np.zeros(x_shape[1:])
        y_means = np.zeros(y_shape[1:])
        y_m2 = np.zeros(y_shape[1:])
        n = 0
        for X, y, _, _ in self.itersamples():
            n += 1
            if X_stats:
                dx = X - X_means
                X_means += dx / n
                X_m2 += dx * (X - X_means)
            if y_stats:
                dy = y - y_means
                y_means += dy / n
                y_m2 += dy * (y - y_means)
        if n < 2:
            X_stds = np.zeros(x_shape[1:])
            y_stds = np.zeros(y_shape[1:])
        else:
            X_stds = np.sqrt(X_m2 / n)
            y_stds = np.sqrt(y_m2 / n)
        if X_stats and not y_stats:
            return X_means, X_stds
        elif y_stats and not X_stats:
            return y_means, y_stds
        elif X_stats and y_stats:
            return X_means, X_stds, y_means, y_stds
        else:
            return tuple()

    def make_tf_dataset(self,
                        batch_size: int = 100,
                        epochs: int = 1,
                        deterministic: bool = False,
                        pad_batches: bool = False):
        try:
            import tensorflow as tf
        except:
            raise ImportError(
                "This method requires TensorFlow to be installed.")

        # Retrieve the first sample so we can determine the dtypes.
        X, y, w, ids = next(self.itersamples())
        dtypes = (tf.as_dtype(X.dtype), tf.as_dtype(y.dtype),
                  tf.as_dtype(w.dtype))
        shapes = (
            tf.TensorShape([None] + list(X.shape)),  # type: ignore
            tf.TensorShape([None] + list(y.shape)),  # type: ignore
            tf.TensorShape([None] + list(w.shape)))  # type: ignore

        # Create a Tensorflow Dataset.
        def gen_data():
            for X, y, w, ids in self.iterbatches(batch_size, epochs,
                                                 deterministic, pad_batches):
                yield (X, y, w)

        return tf.data.Dataset.from_generator(gen_data, dtypes, shapes)

    def make_pytorch_dataset(self,
                             epochs: int = 1,
                             deterministic: bool = False,
                             batch_size: Optional[int] = None):
        raise NotImplementedError()

    def to_dataframe(self) -> pd.DataFrame:
        X = self.X
        y = self.y
        w = self.w
        ids = self.ids
        if len(X.shape) == 1 or X.shape[1] == 1:
            columns = ['X']
        else:
            columns = [f'X{i+1}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=columns)
        if len(y.shape) == 1 or y.shape[1] == 1:
            columns = ['y']
        else:
            columns = [f'y{i+1}' for i in range(y.shape[1])]
        y_df = pd.DataFrame(y, columns=columns)
        if len(w.shape) == 1 or w.shape[1] == 1:
            columns = ['w']
        else:
            columns = [f'w{i+1}' for i in range(w.shape[1])]
        w_df = pd.DataFrame(w, columns=columns)
        ids_df = pd.DataFrame(ids, columns=['ids'])
        return pd.concat([X_df, y_df, w_df, ids_df], axis=1, sort=False)

    @staticmethod
    def from_dataframe(df: pd.DataFrame,
                       X: Optional[OneOrMany[str]] = None,
                       y: Optional[OneOrMany[str]] = None,
                       w: Optional[OneOrMany[str]] = None,
                       ids: Optional[str] = None):
        # Find the X values.
        if X is not None:
            X_val = df[X]
        elif 'X' in df.columns:
            X_val = df['X']
        else:
            columns = []
            i = 1
            while f'X{i}' in df.columns:
                columns.append(f'X{i}')
                i += 1
            X_val = df[columns]
        if len(X_val.shape) == 1:
            X_val = np.expand_dims(X_val, 1)

        # Find the y values.
        if y is not None:
            y_val = df[y]
        elif 'y' in df.columns:
            y_val = df['y']
        else:
            columns = []
            i = 1
            while f'y{i}' in df.columns:
                columns.append(f'y{i}')
                i += 1
            y_val = df[columns]
        if len(y_val.shape) == 1:
            y_val = np.expand_dims(y_val, 1)

        # Find the w values.
        if w is not None:
            w_val = df[w]
        elif 'w' in df.columns:
            w_val = df['w']
        else:
            columns = []
            i = 1
            while f'w{i}' in df.columns:
                columns.append(f'w{i}')
                i += 1
            w_val = df[columns]
        if len(w_val.shape) == 1:
            w_val = np.expand_dims(w_val, 1)

        # Find the ids.
        if ids is not None:
            ids_val = df[ids]
        elif 'ids' in df.columns:
            ids_val = df['ids']
        else:
            ids_val = None
        return NumpyDataset(X_val, y_val, w_val, ids_val)

    def to_csv(self, path: str) -> None:

        columns = []
        X_shape, y_shape, w_shape, id_shape = self.get_shape()
        assert len(
            X_shape) == 2, "dataset's X values should be scalar or 1-D arrays"
        assert len(
            y_shape) == 2, "dataset's y values should be scalar or 1-D arrays"
        if X_shape[1] == 1:
            columns.append('X')
        else:
            columns.extend([f'X{i+1}' for i in range(X_shape[1])])
        if y_shape[1] == 1:
            columns.append('y')
        else:
            columns.extend([f'y{i+1}' for i in range(y_shape[1])])
        if w_shape[1] == 1:
            columns.append('w')
        else:
            columns.extend([f'w{i+1}' for i in range(w_shape[1])])
        columns.append('ids')
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)
            for (x, y, w, ids) in self.itersamples():
                writer.writerow(list(x) + list(y) + list(w) + [ids])
        return None


class NumpyDataset(Dataset):
    def __init__(self,
                 X: ArrayLike,
                 y: Optional[ArrayLike] = None,
                 w: Optional[ArrayLike] = None,
                 ids: Optional[ArrayLike] = None,
                 n_tasks: int = 1) -> None:

        n_samples = np.shape(X)[0]
        if n_samples > 0:
            if y is None:
                # Set labels to be zero, with zero weights
                y = np.zeros((n_samples, n_tasks), np.float32)
                w = np.zeros((n_samples, 1), np.float32)
        if ids is None:
            ids = np.arange(n_samples)
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if w is None:
            if len(y.shape) == 1:
                w = np.ones(y.shape[0], np.float32)
            else:
                w = np.ones((y.shape[0], 1), np.float32)
        if not isinstance(w, np.ndarray):
            w = np.array(w)
        self._X = X
        self._y = y
        self._w = w
        self._ids = np.array(ids, dtype=object)

    def __len__(self) -> int:
        """Get the number of elements in the dataset."""
        return len(self._y)

    def get_shape(self) -> Tuple[Shape, Shape, Shape, Shape]:
        """Get the shape of the dataset.

        Returns four tuples, giving the shape of the X, y, w, and ids arrays.
        """
        return self._X.shape, self._y.shape, self._w.shape, self._ids.shape

    def get_task_names(self) -> np.ndarray:
        """Get the names of the tasks associated with this dataset."""
        if len(self._y.shape) < 2:
            return np.array([0])
        return np.arange(self._y.shape[1])

    @property
    def X(self) -> np.ndarray:
        """Get the X vector for this dataset as a single numpy array."""
        return self._X

    @property
    def y(self) -> np.ndarray:
        """Get the y vector for this dataset as a single numpy array."""
        return self._y

    @property
    def ids(self) -> np.ndarray:
        """Get the ids vector for this dataset as a single numpy array."""
        return self._ids

    @property
    def w(self) -> np.ndarray:
        """Get the weight vector for this dataset as a single numpy array."""
        return self._w

    def iterbatches(self,
                    batch_size: Optional[int] = None,
                    epochs: int = 1,
                    deterministic: bool = False,
                    pad_batches: bool = False) -> Iterator[Batch]:

        def iterate(dataset: NumpyDataset, batch_size: Optional[int],
                    epochs: int, deterministic: bool, pad_batches: bool):
            n_samples = dataset._X.shape[0]
            if deterministic:
                sample_perm = np.arange(n_samples)
            if batch_size is None:
                batch_size = n_samples
            for epoch in range(epochs):
                if not deterministic:
                    sample_perm = np.random.permutation(n_samples)
                batch_idx = 0
                num_batches = math.ceil(n_samples / batch_size)
                while batch_idx < num_batches:
                    start = batch_idx * batch_size
                    end = min(n_samples, (batch_idx + 1) * batch_size)
                    indices = range(start, end)
                    perm_indices = sample_perm[indices]
                    X_batch = dataset._X[perm_indices]
                    y_batch = dataset._y[perm_indices]
                    w_batch = dataset._w[perm_indices]
                    ids_batch = dataset._ids[perm_indices]
                    if pad_batches:
                        (X_batch, y_batch, w_batch,
                         ids_batch) = pad_batch(batch_size, X_batch, y_batch,
                                                w_batch, ids_batch)
                    batch_idx += 1
                    yield (X_batch, y_batch, w_batch, ids_batch)

        return iterate(self, batch_size, epochs, deterministic, pad_batches)

    def itersamples(self) -> Iterator[Batch]:

        n_samples = self._X.shape[0]
        return ((self._X[i], self._y[i], self._w[i], self._ids[i])
                for i in range(n_samples))

    def transform(self, transformer: "dc.trans.Transformer",
                  **args) -> "NumpyDataset":
        newx, newy, neww, newids = transformer.transform_array(
            self._X, self._y, self._w, self._ids)
        return NumpyDataset(newx, newy, neww, newids)

    def select(self,
               indices: Union[Sequence[int], np.ndarray],
               select_dir: Optional[str] = None) -> "NumpyDataset":
       
        X = self.X[indices]
        y = self.y[indices]
        w = self.w[indices]
        ids = self.ids[indices]
        return NumpyDataset(X, y, w, ids)

    def make_pytorch_dataset(self,
                             epochs: int = 1,
                             deterministic: bool = False,
                             batch_size: Optional[int] = None):

        try:
            from deepchem.data.pytorch_datasets import _TorchNumpyDataset
        except:
            raise ImportError("This method requires PyTorch to be installed.")

        pytorch_ds = _TorchNumpyDataset(numpy_dataset=self,
                                        epochs=epochs,
                                        deterministic=deterministic,
                                        batch_size=batch_size)
        return pytorch_ds

    @staticmethod
    def from_DiskDataset(ds: "DiskDataset") -> "NumpyDataset":

        return NumpyDataset(ds.X, ds.y, ds.w, ds.ids)

    def to_json(self, fname: str) -> None:

        d = {
            'X': self.X.tolist(),
            'y': self.y.tolist(),
            'w': self.w.tolist(),
            'ids': self.ids.tolist()
        }
        with open(fname, 'w') as fout:
            json.dump(d, fout)

    @staticmethod
    def from_json(fname: str) -> "NumpyDataset":
        with open(fname) as fin:
            d = json.load(fin)
            return NumpyDataset(d['X'], d['y'], d['w'], d['ids'])

    @staticmethod
    def merge(datasets: Sequence[Dataset]) -> "NumpyDataset":

        X, y, w, ids = datasets[0].X, datasets[0].y, datasets[0].w, datasets[
            0].ids
        for dataset in datasets[1:]:
            X = np.concatenate([X, dataset.X], axis=0)
            y = np.concatenate([y, dataset.y], axis=0)
            w = np.concatenate([w, dataset.w], axis=0)
            ids = np.concatenate(
                [ids, dataset.ids],
                axis=0,
            )
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return NumpyDataset(X, y, w, ids, n_tasks=y.shape[1])


class _Shard(object):

    def __init__(self, X, y, w, ids):
        self.X = X
        self.y = y
        self.w = w
        self.ids = ids


class DiskDataset(Dataset):
    

    def __init__(self, data_dir: str) -> None:

        self.data_dir = data_dir

        logger.info("Loading dataset from disk.")
        tasks, self.metadata_df = self.load_metadata()
        self.tasks = np.array(tasks)
        if len(self.metadata_df.columns) == 4 and list(
                self.metadata_df.columns) == ['ids', 'X', 'y', 'w']:
            logger.info(
                "Detected legacy metatadata on disk. You can upgrade from legacy metadata "
                "to the more efficient current metadata by resharding this dataset "
                "by calling the reshard() method of this object.")
            self.legacy_metadata = True
        elif len(self.metadata_df.columns) == 8 and list(
                self.metadata_df.columns) == [
                    'ids', 'X', 'y', 'w', 'ids_shape', 'X_shape', 'y_shape',
                    'w_shape'
                ]:  # noqa
            self.legacy_metadata = False
        else:
            raise ValueError(
                "Malformed metadata on disk. Metadata must have columns 'ids', 'X', 'y', 'w', "
                "'ids_shape', 'X_shape', 'y_shape', 'w_shape' (or if in legacy metadata format,"
                "columns 'ids', 'X', 'y', 'w')")
        self._cached_shards: Optional[List] = None
        self._memory_cache_size = 20 * (1 << 20)  # 20 MB
        self._cache_used = 0

    @staticmethod
    def create_dataset(shard_generator: Iterable[Batch],
                       data_dir: Optional[str] = None,
                       tasks: Optional[ArrayLike] = None) -> "DiskDataset":

        if data_dir is None:
            data_dir = tempfile.mkdtemp()
        elif not os.path.exists(data_dir):
            os.makedirs(data_dir)

        metadata_rows = []
        time1 = time.time()
        for shard_num, (X, y, w, ids) in enumerate(shard_generator):
            if shard_num == 0:
                if tasks is None and y is not None:
                    # The line here assumes that y generated by shard_generator is a numpy array
                    tasks = np.array([0]) if y.ndim < 2 else np.arange(
                        y.shape[1])
            basename = "shard-%d" % shard_num
            metadata_rows.append(
                DiskDataset.write_data_to_disk(data_dir, basename, X, y, w,
                                               ids))
        metadata_df = DiskDataset._construct_metadata(metadata_rows)
        DiskDataset._save_metadata(metadata_df, data_dir, tasks)
        time2 = time.time()
        logger.info("TIMING: dataset construction took %0.3f s" %
                    (time2 - time1))
        return DiskDataset(data_dir)

    def load_metadata(self) -> Tuple[List[str], pd.DataFrame]:
        """Helper method that loads metadata from disk."""
        try:
            tasks_filename, metadata_filename = self._get_metadata_filename()
            with open(tasks_filename) as fin:
                tasks = json.load(fin)
            metadata_df = pd.read_csv(metadata_filename,
                                      compression='gzip',
                                      dtype=object)
            metadata_df = metadata_df.where((pd.notnull(metadata_df)), None)
            return tasks, metadata_df
        except Exception:
            pass

        # Load obsolete format -> save in new format
        metadata_filename = os.path.join(self.data_dir, "metadata.joblib")
        if os.path.exists(metadata_filename):
            tasks, metadata_df = load_from_disk(metadata_filename)
            del metadata_df['task_names']
            del metadata_df['basename']
            DiskDataset._save_metadata(metadata_df, self.data_dir, tasks)
            return tasks, metadata_df
        raise ValueError(f"No Metadata found in the path {self.data_dir}")

    @staticmethod
    def _save_metadata(metadata_df: pd.DataFrame, data_dir: str,
                       tasks: Optional[ArrayLike]) -> None:

        if tasks is None:
            tasks = []
        elif isinstance(tasks, np.ndarray):
            tasks = tasks.tolist()
        metadata_filename = os.path.join(data_dir, "metadata.csv.gzip")
        tasks_filename = os.path.join(data_dir, "tasks.json")
        with open(tasks_filename, 'w') as fout:
            json.dump(tasks, fout)
        metadata_df.to_csv(metadata_filename, index=False, compression='gzip')

    @staticmethod
    def _construct_metadata(metadata_entries: List) -> pd.DataFrame:
    
        columns = ('ids', 'X', 'y', 'w', 'ids_shape', 'X_shape', 'y_shape',
                   'w_shape')
        metadata_df = pd.DataFrame(metadata_entries, columns=columns)
        return metadata_df

    @staticmethod
    def write_data_to_disk(data_dir: str,
                           basename: str,
                           X: Optional[np.ndarray] = None,
                           y: Optional[np.ndarray] = None,
                           w: Optional[np.ndarray] = None,
                           ids: Optional[np.ndarray] = None) -> List[Any]:

        if X is not None:
            out_X: Optional[str] = "%s-X.npy" % basename
            save_to_disk(X, os.path.join(data_dir, out_X))  # type: ignore
            out_X_shape: Optional[Tuple[int, ...]] = X.shape
        else:
            out_X = None
            out_X_shape = None

        if y is not None:
            out_y: Optional[str] = "%s-y.npy" % basename
            save_to_disk(y, os.path.join(data_dir, out_y))  # type: ignore
            out_y_shape: Optional[Tuple[int, ...]] = y.shape
        else:
            out_y = None
            out_y_shape = None

        if w is not None:
            out_w: Optional[str] = "%s-w.npy" % basename
            save_to_disk(w, os.path.join(data_dir, out_w))  # type: ignore
            out_w_shape: Optional[Tuple[int, ...]] = w.shape
        else:
            out_w = None
            out_w_shape = None

        if ids is not None:
            out_ids: Optional[str] = "%s-ids.npy" % basename
            save_to_disk(ids, os.path.join(data_dir, out_ids))  # type: ignore
            out_ids_shape: Optional[Tuple[int, ...]] = ids.shape
        else:
            out_ids = None
            out_ids_shape = None

        # note that this corresponds to the _construct_metadata column order
        return [
            out_ids, out_X, out_y, out_w, out_ids_shape, out_X_shape,
            out_y_shape, out_w_shape
        ]

    def save_to_disk(self) -> None:
        """Save dataset to disk."""
        DiskDataset._save_metadata(self.metadata_df, self.data_dir, self.tasks)
        self._cached_shards = None

    def move(self,
             new_data_dir: str,
             delete_if_exists: Optional[bool] = True) -> None:

        if delete_if_exists and os.path.isdir(new_data_dir):
            shutil.rmtree(new_data_dir)
        shutil.move(self.data_dir, new_data_dir)
        if delete_if_exists:
            self.data_dir = new_data_dir
        else:
            self.data_dir = os.path.join(new_data_dir,
                                         os.path.basename(self.data_dir))

    def copy(self, new_data_dir: str) -> "DiskDataset":
       
        if os.path.isdir(new_data_dir):
            shutil.rmtree(new_data_dir)
        shutil.copytree(self.data_dir, new_data_dir)
        return DiskDataset(new_data_dir)

    def get_task_names(self) -> np.ndarray:
        """Gets learning tasks associated with this dataset."""
        return self.tasks

    def reshard(self, shard_size: int) -> None:
    
        # Create temp directory to store resharded version
        reshard_dir = tempfile.mkdtemp()
        n_shards = self.get_number_shards()

        # Get correct shapes for y/w
        tasks = self.get_task_names()
        _, y_shape, w_shape, _ = self.get_shape()
        if len(y_shape) == 1:
            y_shape = (len(y_shape), len(tasks))
        if len(w_shape) == 1:
            w_shape = (len(w_shape), len(tasks))

        # Write data in new shards
        def generator():
            X_next = np.zeros((0,) + self.get_data_shape())
            y_next = np.zeros((0,) + y_shape[1:])
            w_next = np.zeros((0,) + w_shape[1:])
            ids_next = np.zeros((0,), dtype=object)
            for shard_num, (X, y, w, ids) in enumerate(self.itershards()):
                logger.info("Resharding shard %d/%d" %
                            (shard_num + 1, n_shards))
                # Handle shapes
                X = np.reshape(X, (len(X),) + self.get_data_shape())
                # Note that this means that DiskDataset resharding currently doesn't
                # work for datasets that aren't regression/classification.
                if y is None:  # datasets without label
                    y = y_next
                    w = w_next
                else:
                    y = np.reshape(y, (len(y),) + y_shape[1:])
                    w = np.reshape(w, (len(w),) + w_shape[1:])
                X_next = np.concatenate([X_next, X], axis=0)
                y_next = np.concatenate([y_next, y], axis=0)
                w_next = np.concatenate([w_next, w], axis=0)
                ids_next = np.concatenate([ids_next, ids])
                while len(X_next) > shard_size:
                    X_batch, X_next = X_next[:shard_size], X_next[shard_size:]
                    y_batch, y_next = y_next[:shard_size], y_next[shard_size:]
                    w_batch, w_next = w_next[:shard_size], w_next[shard_size:]
                    ids_batch, ids_next = ids_next[:shard_size], ids_next[
                        shard_size:]
                    yield (X_batch, y_batch, w_batch, ids_batch)
            # Handle spillover from last shard
            yield (X_next, y_next, w_next, ids_next)

        resharded_dataset = DiskDataset.create_dataset(generator(),
                                                       data_dir=reshard_dir,
                                                       tasks=self.tasks)
        shutil.rmtree(self.data_dir)
        shutil.move(reshard_dir, self.data_dir)
        # Should have updated to non-legacy metadata
        self.legacy_metadata = False
        self.metadata_df = resharded_dataset.metadata_df
        # Note that this resets the cache internally
        self.save_to_disk()

    def get_data_shape(self) -> Shape:
        """Gets array shape of datapoints in this dataset."""
        if not len(self.metadata_df):
            raise ValueError("No data in dataset.")
        if self.legacy_metadata:
            sample_X = load_from_disk(
                os.path.join(self.data_dir,
                             next(self.metadata_df.iterrows())[1]['X']))
            return np.shape(sample_X)[1:]
        else:
            X_shape, _, _, _ = self.get_shape()
            return X_shape[1:]

    def get_shard_size(self) -> int:
        """Gets size of shards on disk."""
        if not len(self.metadata_df):
            raise ValueError("No data in dataset.")
        sample_ids = load_from_disk(
            os.path.join(self.data_dir,
                         next(self.metadata_df.iterrows())[1]['ids']))
        return len(sample_ids)

    def _get_metadata_filename(self) -> Tuple[str, str]:
        """Get standard location for metadata file."""
        metadata_filename = os.path.join(self.data_dir, "metadata.csv.gzip")
        tasks_filename = os.path.join(self.data_dir, "tasks.json")
        return tasks_filename, metadata_filename

    def get_number_shards(self) -> int:
        """Returns the number of shards for this dataset."""
        return self.metadata_df.shape[0]

    def itershards(self) -> Iterator[Batch]:

        return (self.get_shard(i) for i in range(self.get_number_shards()))

    def iterbatches(self,
                    batch_size: Optional[int] = None,
                    epochs: int = 1,
                    deterministic: bool = False,
                    pad_batches: bool = False) -> Iterator[Batch]:

        shard_indices = list(range(self.get_number_shards()))
        return self._iterbatches_from_shards(shard_indices, batch_size, epochs,
                                             deterministic, pad_batches)

    def _iterbatches_from_shards(self,
                                 shard_indices: Sequence[int],
                                 batch_size: Optional[int] = None,
                                 epochs: int = 1,
                                 deterministic: bool = False,
                                 pad_batches: bool = False) -> Iterator[Batch]:
        """Get an object that iterates over batches from a restricted set of shards."""

        def iterate(dataset: DiskDataset, batch_size: Optional[int],
                    epochs: int):
            num_shards = len(shard_indices)
            if deterministic:
                shard_perm = np.arange(num_shards)

            # (ytz): Depending on the application, thread-based pools may be faster
            # than process based pools, since process based pools need to pickle/serialize
            # objects as an extra overhead. Also, as hideously as un-thread safe this looks,
            # we're actually protected by the GIL.
            # mp.dummy aliases ThreadPool to Pool
            pool = Pool(1)

            if batch_size is None:
                num_global_batches = num_shards
            else:
                num_global_batches = math.ceil(dataset.get_shape()[0][0] /
                                               batch_size)

            for epoch in range(epochs):
                if not deterministic:
                    shard_perm = np.random.permutation(num_shards)
                next_shard = pool.apply_async(dataset.get_shard,
                                              (shard_indices[shard_perm[0]],))
                cur_global_batch = 0
                cur_shard = 0
                carry = None

                while cur_global_batch < num_global_batches:

                    X, y, w, ids = next_shard.get()
                    if cur_shard < num_shards - 1:
                        next_shard = pool.apply_async(
                            dataset.get_shard,
                            (shard_indices[shard_perm[cur_shard + 1]],))
                    elif epoch == epochs - 1:
                        pool.close()

                    if carry is not None:
                        X = np.concatenate([carry[0], X], axis=0)
                        if y is not None:
                            y = np.concatenate([carry[1], y], axis=0)
                        if w is not None:
                            w = np.concatenate([carry[2], w], axis=0)
                        ids = np.concatenate([carry[3], ids], axis=0)
                        carry = None

                    n_shard_samples = X.shape[0]
                    cur_local_batch = 0
                    if batch_size is None:
                        shard_batch_size = n_shard_samples
                    else:
                        shard_batch_size = batch_size

                    if n_shard_samples == 0:
                        cur_shard += 1
                        if batch_size is None:
                            cur_global_batch += 1
                        continue

                    num_local_batches = math.ceil(n_shard_samples /
                                                  shard_batch_size)
                    if not deterministic:
                        sample_perm = np.random.permutation(n_shard_samples)
                    else:
                        sample_perm = np.arange(n_shard_samples)

                    while cur_local_batch < num_local_batches:
                        start = cur_local_batch * shard_batch_size
                        end = min(n_shard_samples,
                                  (cur_local_batch + 1) * shard_batch_size)

                        indices = range(start, end)
                        perm_indices = sample_perm[indices]
                        X_b = X[perm_indices]

                        if y is not None:
                            y_b = y[perm_indices]
                        else:
                            y_b = None

                        if w is not None:
                            w_b = w[perm_indices]
                        else:
                            w_b = None

                        ids_b = ids[perm_indices]

                        assert len(X_b) <= shard_batch_size
                        if len(
                                X_b
                        ) < shard_batch_size and cur_shard != num_shards - 1:
                            assert carry is None
                            carry = [X_b, y_b, w_b, ids_b]
                        else:

                            # (ytz): this skips everything except possibly the last shard
                            if pad_batches:
                                (X_b, y_b, w_b,
                                 ids_b) = pad_batch(shard_batch_size, X_b, y_b,
                                                    w_b, ids_b)

                            yield X_b, y_b, w_b, ids_b
                            cur_global_batch += 1
                        cur_local_batch += 1
                    cur_shard += 1

        return iterate(self, batch_size, epochs)

    def itersamples(self) -> Iterator[Batch]:


        def iterate(dataset):
            for (X_shard, y_shard, w_shard, ids_shard) in dataset.itershards():
                n_samples = X_shard.shape[0]
                for i in range(n_samples):

                    def sanitize(elem):
                        if elem is None:
                            return None
                        else:
                            return elem[i]

                    yield map(sanitize, [X_shard, y_shard, w_shard, ids_shard])

        return iterate(self)

    def transform(self,
                  transformer: "dc.trans.Transformer",
                  parallel: bool = False,
                  out_dir: Optional[str] = None,
                  **args) -> "DiskDataset":

        if out_dir is None:
            out_dir = tempfile.mkdtemp()
        tasks = self.get_task_names()
        n_shards = self.get_number_shards()

        time1 = time.time()
        if parallel:
            results = []
            pool = multiprocessing.Pool()
            for i in range(self.get_number_shards()):
                row = self.metadata_df.iloc[i]
                X_file = os.path.join(self.data_dir, row['X'])
                if row['y'] is not None:
                    y_file: Optional[str] = os.path.join(
                        self.data_dir, row['y'])
                else:
                    y_file = None
                if row['w'] is not None:
                    w_file: Optional[str] = os.path.join(
                        self.data_dir, row['w'])
                else:
                    w_file = None
                ids_file = os.path.join(self.data_dir, row['ids'])
                results.append(
                    pool.apply_async(DiskDataset._transform_shard,
                                     (transformer, i, X_file, y_file, w_file,
                                      ids_file, out_dir, tasks)))
            pool.close()
            metadata_rows = [r.get() for r in results]
            metadata_df = DiskDataset._construct_metadata(metadata_rows)
            DiskDataset._save_metadata(metadata_df, out_dir, tasks)
            dataset = DiskDataset(out_dir)
        else:

            def generator():
                for shard_num, row in self.metadata_df.iterrows():
                    logger.info("Transforming shard %d/%d" %
                                (shard_num, n_shards))
                    X, y, w, ids = self.get_shard(shard_num)
                    newx, newy, neww, newids = transformer.transform_array(
                        X, y, w, ids)
                    yield (newx, newy, neww, newids)

            dataset = DiskDataset.create_dataset(generator(),
                                                 data_dir=out_dir,
                                                 tasks=tasks)
        time2 = time.time()
        logger.info("TIMING: transforming took %0.3f s" % (time2 - time1))
        return dataset

    @staticmethod
    def _transform_shard(transformer: "dc.trans.Transformer", shard_num: int,
                         X_file: str, y_file: str, w_file: str, ids_file: str,
                         out_dir: str,
                         tasks: np.ndarray) -> List[Optional[str]]:
        """This is called by transform() to transform a single shard."""
        X = None if X_file is None else np.array(load_from_disk(X_file))
        y = None if y_file is None else np.array(load_from_disk(y_file))
        w = None if w_file is None else np.array(load_from_disk(w_file))
        ids = np.array(load_from_disk(ids_file))
        X, y, w, ids = transformer.transform_array(X, y, w, ids)
        basename = "shard-%d" % shard_num
        return DiskDataset.write_data_to_disk(out_dir, basename, X, y, w, ids)

    def make_pytorch_dataset(self,
                             epochs: int = 1,
                             deterministic: bool = False,
                             batch_size: Optional[int] = None):
        
        try:
            from deepchem.data.pytorch_datasets import _TorchDiskDataset
        except:
            raise ImportError("This method requires PyTorch to be installed.")

        pytorch_ds = _TorchDiskDataset(disk_dataset=self,
                                       epochs=epochs,
                                       deterministic=deterministic,
                                       batch_size=batch_size)
        return pytorch_ds

    @staticmethod
    def from_numpy(X: ArrayLike,
                   y: Optional[ArrayLike] = None,
                   w: Optional[ArrayLike] = None,
                   ids: Optional[ArrayLike] = None,
                   tasks: Optional[ArrayLike] = None,
                   data_dir: Optional[str] = None) -> "DiskDataset":

        # To unify shape handling so from_numpy behaves like NumpyDataset, we just
        # make a NumpyDataset under the hood
        dataset = NumpyDataset(X, y, w, ids)
        if tasks is None:
            tasks = dataset.get_task_names()

        # raw_data = (X, y, w, ids)
        return DiskDataset.create_dataset(
            [(dataset.X, dataset.y, dataset.w, dataset.ids)],
            data_dir=data_dir,
            tasks=tasks)

    @staticmethod
    def merge(datasets: Iterable["Dataset"],
              merge_dir: Optional[str] = None) -> "DiskDataset":

        
        if merge_dir is not None:
            if not os.path.exists(merge_dir):
                os.makedirs(merge_dir)
        else:
            merge_dir = tempfile.mkdtemp()

        # Protect against generator exhaustion
        datasets = list(datasets)

        # This ensures tasks are consistent for all datasets
        tasks = []
        for dataset in datasets:
            try:
                tasks.append(dataset.tasks)  # type: ignore
            except AttributeError:
                pass
        if tasks:
            task_tuples = [tuple(task_list) for task_list in tasks]
            if len(tasks) < len(datasets) or len(set(task_tuples)) > 1:
                raise ValueError(
                    'Cannot merge datasets with different task specifications')
            merge_tasks = tasks[0]
        else:
            merge_tasks = []

        # determine the shard sizes of the datasets to merge
        shard_sizes = []
        for dataset in datasets:
            if hasattr(dataset, 'get_shard_size'):
                shard_sizes.append(dataset.get_shard_size())  # type: ignore
            # otherwise the entire dataset is the "shard size"
            else:
                shard_sizes.append(len(dataset))

        def generator():
            for ind, dataset in enumerate(datasets):
                logger.info("Merging in dataset %d/%d" % (ind, len(datasets)))
                if hasattr(dataset, 'itershards'):
                    for (X, y, w, ids) in dataset.itershards():
                        yield (X, y, w, ids)
                else:
                    yield (dataset.X, dataset.y, dataset.w, dataset.ids)

        merged_dataset = DiskDataset.create_dataset(generator(),
                                                    data_dir=merge_dir,
                                                    tasks=merge_tasks)

        # we must reshard the dataset to have a uniform size
        # choose the smallest shard size
        if len(set(shard_sizes)) > 1:
            merged_dataset.reshard(min(shard_sizes))

        return merged_dataset