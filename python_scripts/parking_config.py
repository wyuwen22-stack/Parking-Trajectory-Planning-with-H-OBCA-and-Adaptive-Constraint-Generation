"""Shared delivery parameters. Restart Python after editing contract.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_config():
    return json.loads((ROOT / 'contract.json').read_text(encoding='utf-8'))


VEHICLE = read_config()['vehicle']
PLANNER = read_config()['planner']
