from setuptools import setup, find_packages
from meteor.version import __version__

#
description = "A tool for automated spectral line identification."

install_requires = [
    "numpy>=1.23",
    "scipy>=1.10",
    "pandas>=2.0",
    "swing-opt",
]

#
setup(
    name='meteor',
    version=__version__,
    author='Yisheng Qiu',
    author_email="hpc_yqiuu@163.com",
    url="https://github.com/yqiuu/meteor",
    description=description,
    install_requires=install_requires,
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            "meteor-config=meteor.scripts:exec_config",
            "meteor-run=meteor.scripts:exec_fit",
            "meteor-modify=meteor.scripts:exec_modify",
            "meteor-identify=meteor.scripts:exec_identify",
        ],
    },
)
