import logging


def get_logger(file_name:str):
    """logger module for logging
    args: file_name (str) : Name of the file being used"""
    logger = logging.getLogger(file_name)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
