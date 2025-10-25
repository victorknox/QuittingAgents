import json
import os
import os.path as osp
import datetime

import pandas as pd

from .my_typing import *
from toolemu.utils.json_utils import RobustJSONEncoder


def create_folder(file_path: str) -> bool:
    """Create folders for the file_path if the folders do not exist.

    Args:
        file_path (str): The path to the file.

    Returns:
        bool: True if the folder is created, False if the folder already exists.
    """
    folder_path = osp.dirname(file_path)
    # Handle case where folder_path is empty (file is in current directory)
    if folder_path and not osp.exists(file_path):
        os.makedirs(folder_path, exist_ok=True)
        return True
    return False


def remove_file(file_path: str) -> None:
    """Remove a file."""
    if osp.exists(file_path):
        os.remove(file_path)


def append_file(
    file_path: str,
    text: str,
    mode: str = "a",
    seperator: str = "\n\n########################\n",
) -> None:
    """Append text to a file."""
    if create_folder(file_path):
        mode = "w"
    with open(file_path, mode=mode) as f:
        f.write(seperator)
        f.write(text)


def append_jsonl(
    file_path: str,
    content: Any,
    mode: str = "a",
) -> None:
    """Append a json object to a jsonl file."""
    if create_folder(file_path):
        mode = "w"
    with open(file_path, mode=mode) as f:
        f.write(json.dumps(content, cls=RobustJSONEncoder) + "\n")


def write_json(content: Any, file_path: str, indent=4) -> None:
    """Write a json object to a json file."""
    create_folder(file_path)
    with open(file_path, "w") as f:
        json.dump(content, f, indent=indent)


def read_json(file_path: str) -> Any:
    with open(file_path, "r") as f:
        return json.load(f)


def read_jsonl(file_path: str) -> List[Any]:
    all_data = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                all_data.append(data)
    return all_data


def read_file(file_path: str) -> Any:
    suffix = file_path.split(".")[-1]
    if suffix == "json":
        return read_json(file_path)
    elif suffix == "jsonl":
        return read_jsonl(file_path)
    elif suffix in ["txt", "log", "md"]:
        with open(file_path, "r") as f:
            return f.readlines()
    elif suffix in ["csv"]:
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unknown file type: {suffix}")


def remove_file_extension(file_path: str) -> str:
    """Remove the file extension suffix of a file path."""
    return ".".join(file_path.split(".")[:-1])


def replace_file_extension(file_path: str, new_extension: str) -> str:
    """Replace the file extension suffix of a file path."""
    return remove_file_extension(file_path) + f".{new_extension}"


def get_run_id():
    """Return a timestamp-based run id."""
    return datetime.datetime.now().strftime("%d%m_%H%M%S")


def get_output_dir_and_prefix(agent_model_name, agent_type, run_id=None, simulator_type="adv_thought", dump_dir="./dumps/trajectories", flat_output=False):
    """
    Returns the output directory and file prefix for storing experiment results.
    If simulator_type is not adv_thought, include it in the file prefix.
    If flat_output is True, use dump_dir as the output directory directly (no subfolders).
    Otherwise, the output directory is dumps/trajectories/{agent_model_name}_{agent_type}/
    """
    if run_id is None:
        run_id = get_run_id()
    dir_name = f"{agent_model_name}_{agent_type}"
    if flat_output:
        output_dir = dump_dir
    else:
        # Only add dir_name if not already present in dump_dir
        if not dump_dir.rstrip('/').endswith(dir_name):
            output_dir = os.path.join(dump_dir, dir_name)
        else:
            output_dir = dump_dir
    if simulator_type == "adv_thought":
        prefix = f"{agent_model_name}_{agent_type}_{run_id}"
    else:
        prefix = f"{simulator_type}_{agent_model_name}_{agent_type}_{run_id}"
    return output_dir, os.path.join(output_dir, prefix)
