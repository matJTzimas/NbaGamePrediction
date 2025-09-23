import yaml
from nba.exception.exception import NbaException
import sys
import os
from nba.logging.logger import logging
import pickle
from typing import Optional
import pandas as pd
from dagshub import get_repo_bucket_client, auth

class Storage:
    def __init__(self, cloud_option: bool):
        self.cloud_option = cloud_option
        self.repo_name = "NbaGamePrediction"
        self.owner = "matJTzimas"

        if self.cloud_option:
            # Remote path (key inside the bucket)
            self.predictions_path = f"{self.repo_name}/predictions.csv"
            auth.add_app_token(os.getenv("DAGSHUB_USER_TOKEN"))
            # repo string must be "owner/repo"
            self.s3 = get_repo_bucket_client(f"{self.owner}/{self.repo_name}", flavor="s3fs")
        else:
            # Local fallback path
            self.predictions_path = "Artifacts/inference/predictions.csv"
            self.s3 = None

    
    def read_csv(self) -> Optional[pd.DataFrame]:
        """Read CSV from local or S3. Return None if file does not exist."""
        if self.cloud_option:
            if not self.s3.exists(self.predictions_path) or self.s3.size(self.predictions_path) == 0:
                return None
            with self.s3.open(self.predictions_path, "rb") as f:
                return pd.read_csv(f)
        else:
            if not os.path.exists(self.predictions_path) or os.path.getsize(self.predictions_path) == 0:
                return None
            return pd.read_csv(self.predictions_path)

    def to_csv(self, data: pd.DataFrame, **kwargs) -> None:
        """Write CSV to local or S3 (overwrite)."""
        df = data.copy()
        if self.cloud_option:
            with self.s3.open(self.predictions_path, "wb") as f:
                df.to_csv(f, index=False, **kwargs)
        else:
            df.to_csv(self.predictions_path, index=False, **kwargs)

    def save_object(self, file_path: str, obj: object) -> None:
        try:
            if self.cloud_option:
                file_name = file_path.split('/')[-1]
                file_path = f"{self.repo_name}/{file_name}"
                with self.s3.open(file_path, "wb") as f:
                    pickle.dump(obj, f)
            else:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "wb") as file_obj:
                    pickle.dump(obj, file_obj)
        except Exception as e:
            raise NbaException(e, sys)

    def load_object(self, file_path: str) -> object:
        try:
            if self.cloud_option:
                file_name = file_path.split('/')[-1]
                file_path = f"{self.repo_name}/{file_name}"
                if not self.s3.exists(file_path):
                    raise Exception(f"The file: {file_path} does not exist in the bucket")
                with self.s3.open(file_path, "rb") as f:
                    return pickle.load(f)
            else:
                if not os.path.exists(file_path):
                    raise Exception(f"The file: {file_path} does not exist")
                with open(file_path, "rb") as file_obj:
                    return pickle.load(file_obj)
        except Exception as e:
            raise NbaException(e, sys)


    def list_root_files(self, details: bool = True):
        """
        List files at the repository root (no subfolders).
        - When cloud_option=True -> lists objects under <repo_name>/ in the DagsHub S3f3 bucket
        - When cloud_option=False -> lists files in the current working directory

        Returns:
        - If details=True: list[dict] with name/path, type, size, mtime (filtered to files)
        - If details=False: list[str] of file paths (filtered to files)
        """
        if self.cloud_option:
            root = self.repo_name  # the repo root inside the bucket
            if not self.s3.exists(root):
                return []
            # Single-level listing
            items = self.s3.ls(root, detail=True)
            # Keep only files
            files = [it for it in items if it.get("type") == "file"]
            if details:
                # Normalize a couple of field names
                out = []
                for it in files:
                    out.append(
                        it.get("name"))
                return out
            else:
                return [it.get("name") for it in files]
        else:
            # Local: list files in CWD (or change to a fixed folder if you prefer)
            base = os.getcwd()
            if not os.path.exists(base):
                return []
            entries = []
            for name in os.listdir(base):
                p = os.path.join(base, name)
                if os.path.isfile(p):
                    if details:
                        st = os.stat(p)
                        entries.append(p)
                    else:
                        entries.append(p)
            return entries

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

    
def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise NbaException(e, sys)
    
def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise NbaException(e, sys)