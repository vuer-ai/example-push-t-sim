from pathlib import Path

from ..flatten_paths import flatten_paths, get_hash_key
import pytest
from unittest.mock import patch


def test_basic_dictionary_creation():
    paths = [
        "/home/user/file1.txt",
        "/var/log/syslog.log",
    ]
    file_dict, _ = flatten_paths(paths)
    assert len(file_dict) == len(paths), "Dictionary should have the same number of keys as paths."


def test_unique_keys():
    paths = [
        "/tmp/some/tempfile.json",
        "/home/user/other_file.csv",
    ]
    file_dict, _ = flatten_paths(paths)
    assert len(set(file_dict.keys())) == len(paths), "All keys should be unique."


def test_multi_extension_filenames():
    paths = [
        "/home/user/complex.file.tar.gz",
        "/var/data/archive.1.2.3.zip",
    ]
    file_dict, _ = flatten_paths(paths)

    complex_path = "/home/user/complex.file.tar.gz"
    archive_path = "/var/data/archive.1.2.3.zip"
    key1 = get_hash_key(complex_path)
    key2 = get_hash_key(archive_path)

    assert file_dict[key1] == complex_path, "Multi-extension filename not handled correctly."
    assert file_dict[key2] == archive_path, "Multi-extension filename not handled correctly."


def test_path_retrieval():
    paths = [
        "/home/user/file1.txt",
        "/var/log/syslog.log",
        "/tmp/some/tempfile.json",
        "/home/user/other_file.csv",
    ]
    file_dict, _ = flatten_paths(paths)

    for path in paths:
        key = get_hash_key(path)
        assert file_dict[key] == path, f"Dictionary entry for key {key} should match original path."


def test_collision_detection():
    """Test that verifies if the flatten_paths function properly detects and handles hash collisions.
    Tests both regular hash collisions and collisions with preserved folder structure."""
    # Test regular hash collision
    paths = ["/home/user/file1.txt", "/home/user/file2.txt"]
    with patch("vuer_mujoco.schemas.utils.flatten_paths.get_hash_key", return_value="collision_hash"):
        with pytest.raises(ValueError, match="Hash collision"):
            flatten_paths(paths)

    # Test collision with preserved folders
    paths = ["/home/user1/file.txt", "/home/user2/file.txt"]
    with patch("vuer_mujoco.schemas.utils.flatten_paths.get_hash_key", return_value="user1/file.txt"):
        with pytest.raises(ValueError, match="Hash collision"):
            # level 0 should cause a collision.
            flatten_paths(paths, levels_to_preserve=0)


def test_folder_structure_preservation():
    """Test that verifies folder structure preservation functionality."""
    paths = ["/home/user/docs/file1.txt", "/home/user/images/file2.png", "/var/log/app/debug.log"]

    # Test with 1 level preservation
    file_dict, _ = flatten_paths(paths, levels_to_preserve=1)
    assert "docs/file1.txt" in file_dict.keys(), "One level folder structure not preserved"
    assert "images/file2.png" in file_dict.keys(), "One level folder structure not preserved"
    assert "app/debug.log" in file_dict.keys(), "One level folder structure not preserved"

    # Test with 2 levels preservation
    file_dict, _ = flatten_paths(paths, levels_to_preserve=2)
    assert "user/docs/file1.txt" in file_dict.keys(), "Two level folder structure not preserved"
    assert "user/images/file2.png" in file_dict.keys(), "Two level folder structure not preserved"
    assert "log/app/debug.log" in file_dict.keys(), "Two level folder structure not preserved"


def test_identical_paths():
    """Test that verifies identical paths do not trigger a hash collision error.
    When the same path is provided multiple times, it should be handled gracefully."""
    paths = ["./file1.txt", f"../__tests__/file1.txt", "../../utils/__tests__/file1.txt"]
    file_dict, hash_lookup_table = flatten_paths(paths)

    bank = set(hash_lookup_table.values())
    print(hash_lookup_table)
    assert len(bank) == 1, "Dictionary should have only one entry for identical paths"
    key = get_hash_key(paths[0])
    assert file_dict[key] == paths[0], "Dictionary should contain the correct path"


def test_path_normalization():
    """Test that verifies paths are normalized correctly before hashing."""
    paths = ["assets/bowl/visual/model.obj", "../schemas/room_xmls/assets/bowl/visual/model.obj"]
    file_dict, _ = flatten_paths(paths)
    assert len(file_dict) == 1, "Dictionary should normalize paths before hashing"
    key = get_hash_key(paths[0])
    assert file_dict[key] in paths, "Dictionary should contain one of the original paths"


def test_relative_absolute_paths():
    """Test that verifies the handling of relative and absolute path combinations."""
    paths = ["/absolute/path/file.txt", "./relative/path/file.txt", "../other/path/file.txt"]
    file_dict, _ = flatten_paths(paths)
    assert len(file_dict) == 3, "Different paths should create different entries"
    for path in paths:
        key = get_hash_key(path)
        assert file_dict[key] == path, f"Dictionary should preserve original path format for {path}"
