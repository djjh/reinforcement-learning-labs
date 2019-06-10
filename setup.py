from os.path import join, dirname, realpath
from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "The Reinforcement Learning Labs repo is designed to work with Python 3.6" \
    + " and greater. Please install it before proceeding."

__version__ = '0.0.0'

setup(
    name='reinforcement-learning-labs',
    py_modules=['reinforcement-learning-labs'],
    version=__version__,#'0.1',
    install_requires=[
        'coloredlogs',
        'cloudpickle==0.5.2',
        'gym[atari,box2d,classic_control]>=0.10.8',
        'ipython',
        'joblib',
        'matplotlib',
        'mpi4py',
        'numpy==1.16.1',  # 1.16.1 is required to load the imbd dataset (for now)
        'pandas',
        'pytest',
        'psutil',
        'scipy',
        'seaborn==0.8.1',
        'tensorflow>=1.13.0',
        'tensorflow-probability>=0.6.0',
        'tqdm',
        'nevergrad'
    ],
    description="Excercises in reinforcement learning.",
    author="Dylan Holmes",
)
