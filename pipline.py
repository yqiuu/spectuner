import sys
import yaml

from optimize_single import main as main_single
from optimize_combine import main as main_combine

if __name__ == "__main__":
    config = yaml.safe_load(open(sys.argv[1]))
    main_single(config)
    main_combine(config)