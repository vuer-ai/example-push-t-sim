import hashlib
import os
from typing import List, Dict


def get_hash_key(file_path: str, hash_length: int = 8) -> str:
    """Generates a short hash key from the given file path."""
    norm_path = os.path.normpath(file_path)
    print(norm_path, flush=True)
    parts = norm_path.split("/")[-1].split(".")
    file_name = parts[0]
    file_ext = ".".join(parts[1:]) if len(parts) > 1 else ""
    hash_obj = hashlib.md5((file_name + norm_path).encode("utf-8"))
    return f"{hash_obj.hexdigest()[:hash_length]}.{file_ext}" if file_ext else hash_obj.hexdigest()[:hash_length]


def flatten_paths(file_paths: List[str], prefix: str = None, levels_to_preserve: int = 0) -> [Dict[str, str], Dict[str, str]]:
    """Creates a dictionary mapping hash keys to file paths.

    Args:
        file_paths: List of file paths to process
        prefix: Optional prefix to add to all hash keys
        levels_to_preserve: Number of directory levels to preserve in the output keys

    Returns:
        Dictionary mapping hash keys to original file paths
    """
    hash_lookup_table = {}
    path_dict = {}

    for path in file_paths:
        parts = path.split("/")
        if levels_to_preserve > 0:
            preserved = "/".join(parts[-(levels_to_preserve + 1) : -1])
            key = f"{preserved}/{parts[-1]}"
        else:
            key = get_hash_key(path)

        # if key in path_dict and os.path.normpath(path_dict[key]) != os.path.normpath(path):
        #     raise ValueError(f"Hash collision detected for key {key} with paths '{path_dict[key]}' and '{path}'")

        if prefix:
            key = f"{prefix}/{key}"

        # todo: reverse the key / value pair.
        path_dict[key] = path
        hash_lookup_table[path] = key

    return path_dict, hash_lookup_table
