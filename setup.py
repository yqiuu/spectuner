from setuptools import setup, find_packages

# Get version
exec(open('spectuner/version.py', 'r').read())

#
description = "A tool for automated line identification of interstellar molecules."

install_requires = [
    "astropy>=6.1.0",
    "h5py>=3.11",
    "numpy>=1.23",
    "numba>=0.60",
    "scipy>=1.10",
    "pandas>=2.0",
    "matplotlib>=3.7",
    "swing-opt>=0.1.2",
    "tqdm>=4.65",
    "torch>=2.5",
]

#
setup(
    name='spectuner',
    version=__version__,
    author='Yisheng Qiu',
    author_email="hpc_yqiuu@163.com",
    url="https://github.com/yqiuu/spectuner",
    license="BSD",
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
