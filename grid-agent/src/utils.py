import yaml
import logger
import functools
import time
from typing import Dict, List, Tuple

logger = logger.get_logger(__name__)

def timer(func):
    """ Print the runtime of the decorated function """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        logger.debug(f"Starting {func.__name__!r}.")
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logger.debug(f"Finished {func.__name__!r} in {run_time:.4f} secs.")
        return value
    return wrapper_timer

def debug(func):
    """ Print the function signature and return value """
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        logger.debug(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        logger.debug(f"{func.__name__!r} returned {value!r}")
        return value
    return wrapper_debug


def read_yaml(config_path: str) -> dict:
    """
    Reads yaml file and returns as a python dict.

    Args:
        config_path (str) : Filepath of yaml file location.

    Returns:
        dict: A dictionary of the yaml filepath parsed in.
    """
    with open(config_path, "r") as f:
        logger.info(f"Config file read in successfully.")
        return yaml.safe_load(f)

