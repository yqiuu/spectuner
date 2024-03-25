from setuptools import setup, find_packages


install_requires = [
    "numpy>=1.23",
    "scipy>=1.10",
    "pandas>=2.0",
]

# Get version
exec(open('meteor/version.py', 'r').read())
#
setup(
    name='meteor',
    version=__version__,
    author='Yisheng Qiu',
    author_email="hpc_yqiuu@163.com",
    install_requires=install_requires,
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            "meteor-run=meteor.optimize.scripts:main",
            "meteor-identify=meteor.identify.scripts:exec_identify"
        ],
    },
)
