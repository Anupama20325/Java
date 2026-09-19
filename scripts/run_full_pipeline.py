#!/usr/bin/env python3
"""Uses the same safe pipeline; pass --dataset-root containing common-schema CSV files."""
from run_demo import run
import argparse
p=argparse.ArgumentParser();p.add_argument('--dataset-root',default='');a=p.parse_args();run(a.dataset_root,False)
