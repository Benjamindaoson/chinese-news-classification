import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from random_forest_common import train_v2


if __name__ == "__main__":
    train_v2()
