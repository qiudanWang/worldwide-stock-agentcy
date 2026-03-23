import os
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def load_yaml(filename):
    path = os.path.join(CONFIG_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_settings():
    return load_yaml("settings.yaml")


def get_data_path(*parts):
    path = os.path.join(DATA_DIR, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
