from typing import Optional
from pathlib import Path

from ..config import save_config, Config
from ..sl_model import SpectralLineDB
from .run_single import run_individual_line_id
from .run_combine import run_combining_line_id


def run_line_identification(config: Config,
                            result_dir: str,
                            sl_db: Optional[SpectralLineDB]=None):
    """Run the line identification algorithm.

    This function runs both the individual and combining line identification
    methods and saves also the config as `config.pickle` in the result
    directory.

    Args:
        config: ``Config`` instance.
        result_dir: Directory to save the results.
        sl_db: Spectral line database. If this is provided, the code will use
            this database instead of the one defined in the config.
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, result_dir/"config.pickle")
    run_individual_line_id(config, result_dir, need_identify=True, sl_db=sl_db)
    run_combining_line_id(config, result_dir, need_identify=True, sl_db=sl_db)