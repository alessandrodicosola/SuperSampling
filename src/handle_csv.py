from typing import List

import pandas as pd
import argparse
from pathlib import Path
from functools import partial
import matplotlib.pyplot as plt

def prepare_csv(files: List[Path], names: List[str]):
    def iterate(files, names):
        for file, name in zip(files, names):
            df = pd.read_csv(file, index_col="Step", parse_dates=['Wall time'],
                             date_parser=partial(pd.to_datetime, unit='s'))
            df.rename(inplace=True, columns={
                "Step": "step",
                "Wall time": "wall_time",
                "Value": "value"
            })
            yield df.add_suffix(f"_{name}")

    return pd.concat(iterate(files, names), axis=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("handle_csv.py")
    parser.add_argument("--alpha", required=False, type=float, default=0.3)
    parser.add_argument("--files", required=True, type=str, nargs="+")
    parser.add_argument("--names", required=True, type=str, nargs="+")
    result = parser.parse_args()

    files = [Path(file) for file in result.files]

    for file in files:
        if not file.exists():
            raise FileNotFoundError(str(file))

    names = result.names

    df = prepare_csv(files=files, names=names)

    cols = [f"value_{name}" for name in names]

    df[cols].ewm(alpha=result.alpha).mean().plot.line(ylim=(0.00001, 0.1))


    plt.show()