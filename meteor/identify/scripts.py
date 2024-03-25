import yaml
from argparse import ArgumentParser

from .identify import identify


def exec_identify():
    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("target", type=str)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    identify(config, args.target)