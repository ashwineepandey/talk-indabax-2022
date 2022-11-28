import logging
from time import strftime

def get_logger(name):
    file_formatter = logging.Formatter('%(asctime)s::%(levelname)s::%(message)s::module:%(module)s::function:%(module)s',
    datefmt='%d-%b-%y %H:%M:%S')
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S')

    timestamp = str(strftime("%Y%m%d_%H%M%S"))

    file_handler = logging.FileHandler(f"../data/logs/{timestamp}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(console_formatter)

    logger = logging.getLogger(name)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    return logger