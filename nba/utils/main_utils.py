import yaml
from nba.exception.exception import NbaException
import sys
import os


def read_yaml_file(file_path_file:str) -> dict: 
    """
        Read yaml file and return the content
    """
    try:
        with open(file_path_file) as file_obj:
            return yaml.safe_load(file_obj)
    except Exception as e:
        raise NbaException(e, sys)

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NbaException(e, sys)

