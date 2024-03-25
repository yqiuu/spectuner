import yaml
from argparse import ArgumentParser

from .run_single import run_single


def main():
    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    run_single(config)