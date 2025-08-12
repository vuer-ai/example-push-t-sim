import os
from contextlib import contextmanager


@contextmanager
def WorkDir(new_directory):
    """Context manager to temporarily switch the current working directory."""
    original_directory = os.getcwd()
    try:
        os.chdir(new_directory)
        yield  # Control goes back to the code inside the `with` block
    finally:
        os.chdir(original_directory)  # Revert back to the original directory


if __name__ == "__main__":
    file_path = "/assets/scene0001/scene.mjcf.xml"

    # Determine the directory containing the file
    new_directory, file_name = os.path.split(file_path)

    # Use the context manager to temporarily set the working directory
    with WorkDir(new_directory):
        print("Temporarily working in:", os.getcwd())  # Prints the new working directory
        # remove_unused_mesh(file_name)  # Call your function with the file name (not the path)

    # Back to the original directory after the `with` block
    print("Working directory reverted to:", os.getcwd())
