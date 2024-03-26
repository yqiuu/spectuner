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
            "meteor-config=meteor.scripts:exec_config",
            "meteor-run=meteor.scripts:exec_fit",
            "meteor-modify=meteor.scripts:exec_modify",
            "meteor-identify=meteor.scripts:exec_identify",
        ],
    },
)
