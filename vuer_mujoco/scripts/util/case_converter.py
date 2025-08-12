def pascal_to_snake(pascal_str):
    """
    Convert a PascalCase string to snake_case.
    Handles acronyms properly by treating consecutive uppercase letters as a single unit.

    The algorithm works as follows:
    1. Start with the first character in lowercase.
    2. Iterate through the rest of the string character by character.
    3. For each character:
       - If it's uppercase, check if it's part of an acronym by examining adjacent characters.
       - If it's in the middle of an acronym (both previous and next characters are uppercase),
         add the lowercase version without an underscore.
       - If it's the start of a word or an acronym, add an underscore followed by the lowercase version.
       - If it's lowercase, just add it as is.
    4. Join all characters together to form the final string.

    This approach correctly handles cases like "EnvSGD" → "env_sgd" and "XMLHttpRequest" → "xml_http_request".

    Args:
        pascal_str (str): The PascalCase string to convert

    Returns:
        str: The snake_case version of the input string
    """
    if not pascal_str:
        return ""

    # Special case handling for strings like "EnvSGD" -> "env_sgd"
    import re

    # Pattern to identify parts of the string that need to be processed
    # This regex finds:
    # 1. Uppercase letter followed by lowercase letters (standard words)
    # 2. Multiple uppercase letters together (acronyms)
    # 3. Lowercase letters
    pattern = r"([A-Z][0-9]+|[A-Z][a-z]+|[A-Z]+(?=[A-Z][a-z]|$)|[a-z]+|\d+)"
    parts = re.findall(pattern, pascal_str)

    # Convert all parts to lowercase and join with underscores
    return "_".join(part.lower() for part in parts)


def snake_to_pascal(snake_str):
    """
    Convert a snake_case string to PascalCase.

    Args:
        snake_str (str): The snake_case string to convert

    Returns:
        str: The PascalCase version of the input string
    """
    if not snake_str:
        return ""

    components = snake_str.split("_")
    return "".join(x.title() for x in components)
