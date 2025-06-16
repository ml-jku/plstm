from pathlib import Path
import os


def request_pytest_filepath(request, filename, base_path: str = "../outputs_tests"):
    outfile_name = f"{request.node.fspath.basename}::{request.node.name}"
    # Optionally replace characters that are not file-friendly
    outfile_name = (
        outfile_name.replace("[", "_")
        .replace("]", "")
        .replace(",", "_")
        .replace(" ", "")
        .replace("/", "_")
        .replace("::", "__")
    )
    output_dir = get_base_path(filename)
    if os.path.isabs(base_path):
        output_dir = Path(base_path)
    else:
        output_dir = Path(os.path.join(output_dir, base_path, get_path_relative_to_base(Path(filename).parent)))
    return output_dir / outfile_name


def get_path_relative_to_base(file_path: str, base_parent_name: str = "tests") -> str:
    """Get the relative directory from `file_path` to the next parent directory
    named `base_parent_name`.

    Args:
        file_path (str): The absolute or relative file path (__file__).
        base_parent_name (str): The name of the parent directory to search for (default: "tests").

    Returns:
        str: The relative path from `file_path` to the `base_parent_name` directory.
             Returns an empty string if `base_parent_name` is not found.
    """
    base_path = get_base_path(file_path, base_parent_name=base_parent_name)
    if base_path:
        return os.path.relpath(os.path.abspath(file_path), base_path)
    return ""  # `base_parent_name` not found


def get_base_path(file_path: str, base_parent_name: str = "tests") -> str:
    """Get the base directory named `base_parent_name` of `file_path`.

    Args:
        file_path (str): The absolute or relative file path (__file__).
        base_parent_name (str): The name of the parent directory to search for (default: "tests").

    Returns:
        str: The absolute path of `base_parent_name`.
             Returns an empty string if `base_parent_name` is not found.
    """
    current_path = os.path.abspath(file_path)
    while current_path:
        current_path, tail = os.path.split(current_path)
        if tail == base_parent_name:
            dir_path = os.path.join(current_path, base_parent_name)
            if os.path.isdir(dir_path):
                return dir_path
        if current_path == os.path.sep:  # Stop at the root directory
            break
    return ""  # `base_parent_name` not found
