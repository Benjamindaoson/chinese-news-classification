import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from day02_common import generate_v2_process_csv


if __name__ == "__main__":
    print(generate_v2_process_csv())
