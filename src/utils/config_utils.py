import pickle
from io import TextIOWrapper
from config import Config


def load_config(sgancfg):
    # Loading the config file
    if type(sgancfg) == str or type(sgancfg) == TextIOWrapper:
        with open(sgancfg, "rb") as f:
            config = pickle.load(f)
    elif issubclass(type(sgancfg), Config):
        config = sgancfg
    else:
        raise TypeError(
            "sgancfg : unknown type. Must pass a path as a string, an opened file or a Config object"
        )
    return config
