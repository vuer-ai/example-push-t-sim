# Function to walk through leaves
def walk_leaves(element, fn):
    fn(element)
    for child in element:
        walk_leaves(child, fn)


class Whitener:
    @staticmethod
    def __call__(xml_string: str) -> str:
        """
        Minimizes an XML string using lxml by removing unnecessary whitespace and
        collapsing long empty strings between angle brackets.

        Args:
            xml_string (str): The original XML string.

        Returns:
            str: The minimized XML string.
        """
        text = xml_string.split("\n")
        return "".join([line.strip() for line in text])

    def __ror__(self, s: str) -> str:
        return self(s)


whiten = Whitener()

if __name__ == "__main__":
    # Example usage
    xml = """
    <root>
        <person>
            <name>John Doe</name>
            <age>30</age>
        </person>
    </root>
    """

    minimized = whiten(xml)
    print(minimized)
    assert minimized == "<root><person><name>John Doe</name><age>30</age></person></root>"
