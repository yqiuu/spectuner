import yaml
from argparse import ArgumentParser

from .run_identify import run_identify


def identify():
    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("target", type=str)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    run_identify(config, args.target)