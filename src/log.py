import datetime
from logging import INFO, FileHandler, Formatter, Logger, StreamHandler, getLogger
from pprint import pformat
from typing import Any, Final, Sequence

import numpy as np

FORMAT: Final[str] = "[%(levelname)s] %(asctime)s - %(pathname)s : %(lineno)d: %(funcName)s: %(message)s"

logger = getLogger(__name__)


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


def calc_duration_from(strftme: str) -> str:
    """calculate duration from strftme

    Args:
        strftme (str): start time

    Returns:
        str: duration
    """
    start = datetime.datetime.strptime(strftme, "%Y%m%d-%H:%M:%S")
    now = datetime.datetime.now(datetime.timezone.utc)
    duration = now.timestamp() - start.timestamp()
    duration_sec = int(duration)
    duration_min = duration_sec // 60
    return f"{duration_min} [min], {duration_sec} [sec]"


def info_stats(cfg: dict[str, Any], scores: Sequence[float], called_time: str, commit_hash: str) -> None:
    score_mean = np.mean(scores)
    score_std = np.std(scores)
    logger.info(
        "\n===============================================================\n"
        + " End of Training\n"
        + f" Scores: {scores}\n"
        + f" Mean: {score_mean} +/- {score_std}\n"
        + f" Called Time: {called_time}\n"
        + f" DURATION: {calc_duration_from(called_time)}\n"
        + f" COMMIT_HASH: {commit_hash}\n"
        + f" CONFIG: {pformat(cfg)}\n"
        + "===============================================================\n"
    )
