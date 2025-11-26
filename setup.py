from setuptools import setup, find_packages

# Get version
exec(open('spectuner/version.py', 'r').read())

#
description = "A tool for automated spectral line analysis of instellar molecules."

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
    python_requires='>=3.10',
    packages=find_packages(),
    package_data={
        'spectuner': [
            'config/templates/*.yml',
            'ai/normalizations_v1.yml',
        ],
    },
)
