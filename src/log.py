import datetime
from logging import INFO, FileHandler, Formatter, Logger, StreamHandler, getLogger
from typing import Final

FORMAT: Final[str] = "[%(levelname)s] %(asctime)s - %(pathname)s : %(lineno)d: %(funcName)s: %(message)s"


def get_root_logger(level: int = INFO) -> Logger:
    """get root logger

    Args:
        level (int, optional): log level. Defaults to 20 (is equal to INFO).

    Returns:
        Logger: root logger
    """
    logger = getLogger()
    logger.setLevel(level)

    handler = StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(Formatter(FORMAT))

    logger.addHandler(handler)
    return logger


def attach_file_handler(root_logger: Logger, log_fname: str, level: int = 20) -> None:
    """attach file handler to root logger

    Args:
        root_logger (Logger): root logger
        log_fname (str): log file name
        level (int, optional): log level. Defaults to 20 (is equal to INFO).

    """
    handler = FileHandler(log_fname)
    handler.setLevel(level)
    handler.setFormatter(Formatter(FORMAT))

    root_logger.addHandler(handler)


def get_called_time() -> str:
    """get called time (UTC+9 japan)

    Returns:
        str: called time
    """
    now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=9)
    return now.strftime("%Y%m%d-%H:%M:%S")
