from pathlib import Path

import numpy as np
from filelock import FileLock


def dump_memmap(data, file_path):
    """We save the metadata in a `.meta` file in the format dtype:shape."""
    # Create a memmap array and write data
    shape = data.shape
    dtype = data.dtype  # Use the input data's dtype

    if not np.issubdtype(dtype, np.number):
        raise ValueError(f"Data type must be numeric, but received {dtype}")

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_file = file_path.with_suffix(file_path.suffix + ".meta")
    lock_file = file_path.with_suffix(file_path.suffix + ".lock")
    lock = FileLock(lock_file)

    with lock:
        # Avoid rewriting if file already exists
        if metadata_file.exists():
            print(f"File {metadata_file} already exists. Skipping write.")
        else:
            # Save metadata in dtype:shape format
            with open(metadata_file, "w") as meta:
                meta.write(f"{dtype.name}:{shape}\n")

        if file_path.exists() :
            print(f"File {file_path} already exists. Skipping write.")
            return

        # Save data to memmap
        memmap_array = np.memmap(file_path, dtype=dtype, mode="w+", shape=shape)
        memmap_array[:] = data[:]
        memmap_array.flush()
        del memmap_array  # Explicitly release the memmap


def load_memmap(file_path):
    """Load the metadata from `.meta` file in dtype:shape format and then load the memmap."""
    # Check for the metadata file
    metadata_file = f"{file_path}.meta"
    if not Path(metadata_file).exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    # Read and parse the metadata in dtype:shape format
    with open(metadata_file, "r") as meta:
        metadata = meta.readline().strip()
    dtype_str, shape_str = metadata.split(":")
    dtype = np.dtype(dtype_str)
    shape = tuple(map(int, shape_str.strip("()").split(", ")))  # Parse shape

    # Load the memmap array
    memmap_array = np.memmap(file_path, dtype=dtype, mode="r", shape=shape)
    # print("loaded", memmap_array.dtype, memmap_array.shape, "from", file_path)
    return memmap_array
