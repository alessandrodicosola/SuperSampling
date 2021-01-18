import pandas as pd
import argparse
from pathlib import Path
from functools import partial


def prepare_csv(filepath):
    df = pd.read_csv(filepath, index_col="Step", parse_dates=['Wall time'],
                     date_parser=partial(pd.to_datetime, unit='s'))
    df.rename(inplace=True, columns={
        "Step": "step",
        "Wall time": "wall time",
        "Value": "value"
    })
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser("handle_csv.py")
    parser.add_argument("filepath", type=str)
    result = parser.parse_args()
    file = Path(result.filepath)

    if not file.exists():
        raise FileNotFoundError(str(file))

    df = prepare_csv(file)

    df.plot.line()
