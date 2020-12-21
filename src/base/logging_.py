import logging
import sys
from pathlib import Path


########################################################################################################################
def disable_up_to(level: int):
    logging.disable(level - 10)


def get(name: str):
    debug_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    streamHandler = logging.StreamHandler(sys.stdout)
    streamHandler.setLevel(logging.DEBUG)
    streamHandler.setFormatter(debug_formatter)

    mylog = logging.getLogger(name)
    mylog.addHandler(streamHandler)

    return mylog


########################################################################################################################

disable_up_to(logging.NOTSET)
