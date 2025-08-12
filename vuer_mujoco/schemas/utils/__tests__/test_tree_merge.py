from xml.etree.ElementTree import tostring

from vuer_mujoco.schemas.utils import minimize_many
from vuer_mujoco.schemas.utils import merge_trees, merge_many


# Test merging two XML strings with nested tags and overlapping keys.
def test_tree_merge_strings_with_nested_tags():
    xml_string1 = """
    <root>
        <option key="1">Value1</option>
        <option key="3">this is different</option>
        <assets>
            <asset key="1">Asset1</asset>
        </assets>
    </root>
    """

    xml_string2 = """
    <root>
        <option key="1">Value1</option>
        <option key="2">Value2</option>
        <assets>
            <asset key="2">Asset2</asset>
            <asset key="3">Asset3</asset>
        </assets>
    </root>
    """

    expected_output = """
    <root>
        <option key="1">Value1</option>
        <option key="3">this is different</option>
        <assets>
            <asset key="1">Asset1</asset>
            <asset key="2">Asset2</asset>
            <asset key="3">Asset3</asset>
        </assets>
        <option key="2">Value2</option>
    </root>
    """


    merged_tree = merge_trees(xml_string1, xml_string2)

    def clean_xml(xml):
        return minimize_many(tostring(xml).decode())

    merged_string = clean_xml(merged_tree)

    print("")
    print(">>>", merged_string)
    print("==>", expected_output | minimize_many)

    assert merged_string == minimize_many(expected_output), f"Merged XML did not match expected output. Got: {merged_string}"


def test_merge_many():
    xml_string1 = """
    <assets>
        <item key="1">Item1</item>
    </assets>
    <options name="some-variable">10</options>
    <assets>
        <item key="2">Item2</item>
    </assets>
    """

    xml_string2 = """
    <assets>
        <item key="3">Item3</item>
    </assets>
    <options name="some-variable">10</options>
    <assets>
        <item key="4">Item4</item>
    </assets>
    """

    expected_output = """
    <assets>
        <item key="1">Item1</item>
        <item key="2">Item2</item>
        <item key="3">Item3</item>
        <item key="4">Item4</item>
    </assets>
    <options name="some-variable">10</options>
    """

    merged_xml = merge_many(xml_string1, xml_string2)

    merged_string = merged_xml | minimize_many

    # print("")
    # print(">>>", merged_string)
    # print("==>", minimize_many(expected_output))

    assert merged_string == minimize_many(expected_output), f"Merged XML did not match expected output. Got: {merged_string}"
