"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='imfpredictpy',
    version='0.0.2',
    description='Functions and scripts to analyse and predict currency performance',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/benchiverton/ImfPredictPy',
    packages=find_packages(),
    python_requires='>=3.7, <3.7.9',
    install_requires=[
        'numpy',
        'torch',
        'matplotlib'
    ],
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    }
)
