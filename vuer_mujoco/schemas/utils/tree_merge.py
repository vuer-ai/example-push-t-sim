from copy import deepcopy
from typing import Callable

from lxml import etree


# Function to walk through leaves


def walk_leaves(element: etree.Element, fn: Callable[[etree.Element], None]) -> None:
    fn(element)
    for child in element:
        walk_leaves(child, fn)


def _preprocess(xml_string: str) -> etree.Element:
    parser = etree.XMLParser(remove_blank_text=True)

    xml_string = xml_string.strip()
    root = etree.XML(xml_string, parser)

    def strip_whitespace(element: etree.Element):
        if element.text:
            element.text = element.text.strip() or None
        if element.tail:
            element.tail = element.tail.strip() or None

    walk_leaves(root, strip_whitespace)

    return root


def HASH_FN(e: etree.Element) -> str:
    # this means it is a comment component.
    if not isinstance(e.tag, str): #.func_name == 'Comment':
        return str(e)

    k = e.attrib.get("key", "")
    n = e.attrib.get("name", "")
    c = e.attrib.get("class", "")
    return f"<{e.tag}[key:{k},name:{n},class:{c}]/>"


def merge_trees(
    *xml_strings: str,
    hash_fn: Callable[[etree.Element], str] = HASH_FN,
) -> etree.Element:
    """
    Merges two XML strings, removing identical elements.

    Args:
        xml_string1 (str): The first XML string.
        xml_string2 (str): The second XML string.

    Returns:
        ElementTree: A new XML tree containing the merged elements with duplicates removed.
    """

    roots = [_preprocess(s) for s in xml_strings if s and s.strip()]
    if not roots:
        return None

    def merge_children(*nodes: etree.Element) -> etree.Element:
        """
        Recursively merge children of node1 and node2, removing duplicates.

        At each level, identify children that are the same, and then merge their children.

        # Steps for merging two XML trees:
        # 1. Validate that the root elements of both trees have the same tag and aligned attributes.
        # 2. Generate a hash for each child node in both trees to facilitate duplicate detection.
        # 3. Merge nodes with identical hashes by recursively applying this function to their children.
        # 4. Append unique child nodes from node2 into node1 without modification.
        # 5. If applicable, merge attributes of nodes where hashes match.
        """
        if not nodes:
            return None

        first = nodes[0]
        first = deepcopy(first)

        has_seen = {}

        new_root = deepcopy(first)
        try:
            new_root[:] = []
        except TypeError:
            return new_root

        for node in nodes:
            for child in node:
                h = hash_fn(child)
                if h in has_seen:
                    # print("merge child", h)
                    has_seen[h] = merge_children(has_seen[h], child)
                else:
                    # print("Adding new child", h)
                    child_copy = deepcopy(child)
                    has_seen[h] = child_copy

        new_root[:] = [*has_seen.values()]

        return new_root

    return merge_children(*roots)


def merge_many(
    *xml_strings: str,
    hash_fn: Callable[[etree.Element], str] = HASH_FN,
) -> str:
    """
    Merge two XML strings into a unified structure, handling lists of elements.

    This function takes two XML string inputs, wraps each of them inside
    a specific parent tag called `<merge_many>`, and then merges the two
    modified XML strings into a single XML structure. The resulting string
    excludes the wrapping `<merge_many>` tags and removes duplicate nodes.

    Parameters:
        xml_string1 (str): The first XML string to merge.
        xml_string2 (str): The second XML string to merge.
        ...

    Returns:
        str: The merged XML string, with the wrapping `<merge_many>` tags removed.
    """

    xml_strings = [f"<merge_many>{s}</merge_many>" for s in xml_strings if s and s.strip()]

    if not xml_strings:
        return ""

    merged_tree = merge_trees(*xml_strings, hash_fn=hash_fn)

    return etree.tostring(merged_tree, encoding="unicode", pretty_print=False)[12:-13]
