import os
import pathlib

IS_KAGGLE = os.getenv("KAGGLE") is not None
COMPE_NAME = ""

ROOT = pathlib.Path("/kaggle") if IS_KAGGLE else pathlib.Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
DATA_DIR = INPUT_DIR / COMPE_NAME
