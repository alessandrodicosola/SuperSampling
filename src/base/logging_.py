import logging
import sys
from pathlib import Path


########################################################################################################################
def disable_up_to(level: int):
    logging.disable(level - 10)


def get(name: str, error_log_dir: Path = None):
    debug_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    error_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    streamHandler = logging.StreamHandler(sys.stdout)
    streamHandler.setLevel(logging.DEBUG)
    streamHandler.setFormatter(debug_formatter)

    log_path = error_log_dir / f"{name}.error.log" if error_log_dir is not None else None
    fileHander = logging.FileHandler(log_path, mode="w") if log_path is not None else logging.StreamHandler(
        sys.stderr)
    fileHander.setLevel(logging.ERROR + logging.CRITICAL)
    fileHander.setFormatter(error_formatter)

    mylog = logging.getLogger(name)
    mylog.addHandler(streamHandler)
    mylog.addHandler(streamHandler)

    return mylog


########################################################################################################################

disable_up_to(logging.DEBUG)
