import inspect
import logging
import os

from utils import get_model_name


def get_logger(directory):
    # Get the name of the calling module
    caller_frame_record = inspect.stack()[1]  # 0 represents this line
    # 1 represents line at caller
    frame = caller_frame_record[0]
    info = inspect.getframeinfo(frame)
    module_name = os.path.basename(info.filename).replace('.py', '')

    # Create a custom logger
    logger = logging.getLogger(module_name)

    # Set level of logger
    logger.setLevel(logging.DEBUG)

    log_file_path = os.path.join(directory, "log.txt")

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file_path, mode='w')
    c_handler.setLevel(logging.WARNING)
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger
