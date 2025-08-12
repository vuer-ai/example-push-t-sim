class MetaFile(type):
    def __matmul__(cls, filepath: str):
        """
        Opens the given file in read mode and returns its content.

        Args:
            filepath (str): Path to the file to be opened.

        Returns:
            str: Contents of the file as a string.
        """
        try:
            with open(filepath, "r") as f:
                return f.read()
        except Exception as e:
            print(f"Error opening file: {e}")
            return ""


class File(metaclass=MetaFile):
    pass


class Prettify:
    def __call__(self, xml_string: str):
        """
        prettyfies the given XML string.
        """
        try:
            from xml.dom.minidom import parseString

            xml = parseString(xml_string)
            pretty_xml = xml.toprettyxml(indent="  ")
            return "\n".join([line for line in pretty_xml.splitlines() if line.strip()])
        except Exception as e:
            print(f"Error prettyfying XML: {e}")
            return xml_string

    def __ror__(self, *args):
        return self(*args)


def Read(path: str):
    """
    Read a file, and pipe the contents to the next function.

    Args:
        path (str): Path to the file to be saved.
    """

    try:
        with open(path, "r") as f:
            return f.read()

    except Exception as e:
        print(f"Error reading file: {e}")


class Save:
    def __init__(self, path: str, append=False):
        self.path = path
        self.append = append

    def __call__(self, xml_string: str):
        """
        Saves the given XML string to the given file.

        Args:
            path (str): Path to the file to be saved.
            xml_string (str): The XML string to be saved.
        """
        try:
            mode = "a" if self.append else "w+"

            with open(self.path, mode) as f:
                f.write(xml_string)

        except Exception as e:
            print(f"Error saving file: {e}")

    def __ror__(self, *args):
        return self(*args)
