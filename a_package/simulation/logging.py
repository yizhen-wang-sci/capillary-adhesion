import sys
import logging


def reset_logging():
    """
    Config the logger such that logging.info(...) works like print(...)
    """
    root_logger = logging.getLogger()

    # clear any existing handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # config logging to console as if calling print(...)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(console_handler)

    # set logging level
    root_logger.setLevel(logging.INFO)


def switch_log_file(log_file):
    root_logger = logging.getLogger()

    # remove all existing file handler
    for h in list(root_logger.handlers):
        if isinstance(h, logging.FileHandler):
            root_logger.removeHandler(h)
            h.close()

    # add its own file hander
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
    root_logger.addHandler(file_handler)
