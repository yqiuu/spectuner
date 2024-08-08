from setuptools import setup, find_packages
from spectuner.version import __version__

#
description = "A tool for automated line identification of interstellar molecules."

install_requires = [
    "numpy>=1.23",
    "scipy>=1.10",
    "pandas>=2.0",
    "swing-opt",
]

#
setup(
    name='spectuner',
    version=__version__,
    author='Yisheng Qiu',
    author_email="hpc_yqiuu@163.com",
    url="https://github.com/yqiuu/spectuner",
    description=description,
    install_requires=install_requires,
    packages=find_packages(),
    package_data={
        'spectuner': ['config/templates/*.yml'],
    },
    entry_points={
        'console_scripts': [
            "spectuner-config=spectuner.scripts:exec_config",
            "spectuner-run=spectuner.scripts:exec_fit",
            "spectuner-modify=spectuner.scripts:exec_modify",
            "spectuner-identify=spectuner.scripts:exec_identify",
        ],
    },
)
