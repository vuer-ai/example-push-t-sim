import pytest
import numpy as np
from lucidxr.learning.memmap_cache import dump_memmap, load_memmap


@pytest.fixture
def random_tmp_file(tmp_path):
    """Fixture to create a unique random temporary file path."""
    return tmp_path / f"test_memmap.data"


def test_dump_and_load_memmap(random_tmp_file):
    """Test that the dumped data is correctly loaded as a memmap."""
    # Create test data
    data = np.random.rand(100, 100)  # Example numeric data

    # Dump the data
    dump_memmap(data, random_tmp_file)

    # Load the data
    loaded_memmap = load_memmap(random_tmp_file)

    # Ensure the loaded memmap matches original data
    assert isinstance(loaded_memmap, np.memmap)
    assert loaded_memmap.shape == data.shape
    assert loaded_memmap.dtype == data.dtype
    assert np.allclose(data, loaded_memmap)  # Compare numerical values


def test_dump_memmap_creates_file(random_tmp_file):
    """Test that dump_memmap creates a file on disk."""

    print(random_tmp_file)

    data = np.random.rand(10, 10)
    dump_memmap(data, random_tmp_file)
    assert random_tmp_file.exists()  # Check if the file was created


def test_non_numeric_dtype_error(random_tmp_file):
    """Test that dump_memmap raises ValueError for non-numeric data."""
    non_numeric_data = np.array([["a", "b"], ["c", "d"]], dtype=object)

    with pytest.raises(ValueError, match="Data type must be numeric"):
        dump_memmap(non_numeric_data, random_tmp_file)


def test_load_memmap_invalid_file():
    """Test that loading a non-existent file raises an appropriate error."""
    with pytest.raises(OSError):
        load_memmap("non_existent_file.npy")


def test_dump_memmap_overwrite(random_tmp_file):
    """Test that dumping data overwrites existing memmap."""
    # First data dump
    data1 = np.zeros((5, 5), dtype=np.float32)
    dump_memmap(data1, random_tmp_file)

    # Verify contents with first dump
    memmap1 = load_memmap(random_tmp_file)
    assert np.allclose(memmap1, data1)

    # Second data dump (with overwrite)
    data2 = np.ones((5, 5), dtype=np.float32)
    dump_memmap(data2, random_tmp_file)

    # Verify contents after overwrite
    memmap2 = load_memmap(random_tmp_file)
    assert np.allclose(memmap2, data2)


def test_memmap_written_to_disk(random_tmp_file):
    """Test that memmap data is correctly flushed to disk."""
    data = np.random.rand(10, 10)
    dump_memmap(data, random_tmp_file)

    # Check the file size on disk
    expected_size = np.prod(data.shape) * data.dtype.itemsize
    actual_size = random_tmp_file.stat().st_size

    assert expected_size == actual_size


def test_partial_read_memmap(random_tmp_file):
    """Test partial reading of a memmap after loading."""
    data = np.random.rand(100, 100)
    dump_memmap(data, random_tmp_file)

    memmap_array = load_memmap(random_tmp_file)

    # Access a slice of the memmap and validate its content
    partial_data = data[10:20, 10:20]
    assert np.allclose(memmap_array[10:20, 10:20], partial_data)
