import re

# Regular expression to match file="..."
FILE_REGEX = re.compile(r'file="([^"]+)"')


def get_paths(text):
    """
    Extracts all file paths from the given text.

    Args:
        text (str): The text containing the file paths.

    Returns:
        list: List of file paths found in the text.
    """
    return FILE_REGEX.findall(text)


def path_rewrite(text, hash_lookup_table):
    """
    Rewrites file paths in the given text based on hash_mapping.

    Args:
        text (str): The text containing the file paths.
        hash_lookup_table (dict): Dictionary mapping original paths to hash keys.

    Returns:
        str: Text with file paths rewritten.
    """

    def replace(match):
        original_path = match.group(1)
        # Find the corresponding hash key for this path
        path = hash_lookup_table.get(original_path, None)
        if path is None:
            # return the original string
            return match.group(0)
        else:
            return f'file="{path}"'

    return FILE_REGEX.sub(replace, text)
