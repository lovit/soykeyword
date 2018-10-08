import sys
sys.path.insert(0, './')
import soykeyword

import setuptools
from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="soykeyword",
    version=soykeyword.__version__,
    author=soykeyword.__author__,
    author_email='soy.lovit@gmail.com',
    description="Unsupervised Keyword Extracters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/lovit/soykeyword',
    packages=setuptools.find_packages(),
    install_requires=["numpy>=1.12.0", "scikit-learn>=0.18.0", "psutil>=5.0.1"],
    classifiers=(
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ),
)