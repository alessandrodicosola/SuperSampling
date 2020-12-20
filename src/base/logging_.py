import logging
import sys
from pathlib import Path

def disable_up_to(level: int):
    logging.disable(level - 10)

def get(name: str, log_dir: Path):
    debug_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    error_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    streamHandler = logging.StreamHandler(sys.stdout)
    streamHandler.setLevel(logging.DEBUG)
    streamHandler.setFormatter(debug_formatter)

    log_path = log_dir / f"{name}.log"
    fileHander = logging.FileHandler(log_path)
    fileHander.setLevel(logging.ERROR + logging.CRITICAL)
    fileHander.setFormatter(error_formatter)

    mylog = logging.getLogger(name)
    mylog.addHandler(streamHandler)
    mylog.addHandler(streamHandler)

    return mylog

####################

disable_up_to(logging.DEBUG)

