from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_V2 = ROOT / "01-data" / "v2"
TRAIN_TXT = DATA_V2 / "train_v2.txt"
DEV_TXT = DATA_V2 / "dev_v2.txt"
TEST_TXT = DATA_V2 / "test_v2.txt"
PROCESS_TRAIN = DATA_V2 / "process_train_v2.csv"
PROCESS_DEV = DATA_V2 / "process_dev_v2.csv"
PROCESS_TEST = DATA_V2 / "process_test_v2.csv"
