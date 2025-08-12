from vuer_mujoco.schemas.utils.path_rewrite import path_rewrite, get_paths


import pytest


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ('Here are files: file="path/to/file1.txt" and file="another/path/file2.jpg"', ["path/to/file1.txt", "another/path/file2.jpg"]),
        ("No files here", []),
        ('One file="single/path.pdf"', ["single/path.pdf"]),
        ('File with spaces file="path/to my/file.txt"', ["path/to my/file.txt"]),
        ('File with special chars file="path/@#$/file.txt"', ["path/@#$/file.txt"]),
        ('Multiple identical files file="same.txt" file="same.txt"', ["same.txt", "same.txt"]),
    ],
)
def test_get_paths(test_input, expected):
    """Test the get_paths function with various input scenarios.

    Tests include:
    - Multiple files in text
    - No files in text
    - Single file in text
    - Paths with spaces
    - Paths with special characters
    - Duplicate file paths
    """
    assert get_paths(test_input) == expected


def test_path_rewrite():
    """Test the path_rewrite function for various file path replacement scenarios."""
    test_cases = [
        {
            "text": 'Here is a reference file="images/photo.png" and another file="docs/report.pdf".',
            "hash_mapping": {
                "images/photo.png": "abc123.png",
                "docs/report.pdf": "def456.pdf",
            },
            "expected": 'Here is a reference file="abc123.png" and another file="def456.pdf".',
        },
        {"text": "No matching files here.", "hash_mapping": {"abc123.txt": "test.txt"}, "expected": "No matching files here."},
        {
            "text": 'One matching file="docs/report.pdf" and one non-matching file="unknown/file.jpg".',
            "hash_mapping": {"docs/report.pdf": "def456.pdf"},
            "expected": 'One matching file="def456.pdf" and one non-matching file="unknown/file.jpg".',
        },
    ]

    for case in test_cases:
        assert path_rewrite(case["text"], case["hash_mapping"]) == case["expected"]


def test_path_rewrite_2():
    text = 'Here is a reference file="images/photo.png" and another file="docs/report.pdf".'
    hash_mapping = {
        "images/photo.png": "abc123.png",
        "docs/report.pdf": "def456.pdf",
    }

    expected_output = 'Here is a reference file="abc123.png" and another file="def456.pdf".'
    assert path_rewrite(text, hash_mapping) == expected_output

    # Test no replacement
    text_no_match = "No matching files here."
    assert path_rewrite(text_no_match, hash_mapping) == text_no_match

    # Test partial replacement
    text_partial = 'One matching file="docs/report.pdf" and one non-matching file="unknown/file.jpg".'
    expected_partial_output = 'One matching file="def456.pdf" and one non-matching file="unknown/file.jpg".'
    assert path_rewrite(text_partial, hash_mapping) == expected_partial_output


def test_path_rewrite_repeated_strings():
    # Test repeated strings
    text_repeated = 'Repeated file="docs/report.pdf" and again file="docs/report.pdf".'
    hash_mapping = {
        "images/photo.png": "abc123.png",
        "docs/report.pdf": "def456.pdf",
    }
    expected_repeated = 'Repeated file="def456.pdf" and again file="def456.pdf".'
    assert path_rewrite(text_repeated, hash_mapping) == expected_repeated


def test_path_rewrite_empty_inputs():
    """Test path_rewrite function with empty inputs."""
    assert path_rewrite("", {}) == ""
    assert path_rewrite('file="test.txt"', {}) == 'file="test.txt"'
    assert path_rewrite("", {"path": "hash"}) == ""


def test_path_rewrite_invalid_inputs():
    """Test path_rewrite function with invalid inputs."""
    with pytest.raises(TypeError):
        path_rewrite(None, {})
    # with pytest.raises(TypeError):
    #     path_rewrite("text", None)
