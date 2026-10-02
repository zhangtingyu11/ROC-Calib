"""Download the CARLA example (standard library only)."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from roc_calib.demo import download

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1] / "demo/carla")
download(parser.parse_args().destination)
