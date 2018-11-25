from setuptools import setup, find_packages

requirements = [
    'torch',
    'numpy',
    'scipy',
    'matplotlib',
    'scikit-learn',
    'pyyaml'
]

VERSION = '0.0.1'

setup(
    name='extreme_classification',
    version=VERSION,
    url='https://github.com/vishwakftw/extreme-classification',
    description='Python module designed for extreme classification tasks',

    packages=find_packages(),

    zip_safe=True,
    install_requires=requirements
)
