# install using 'pip install -e .'
import os
from setuptools import setup

setup(name='grasp_baseline',
      packages=['grasp_baseline'],
      package_dir={'grasp_baseline': 'grasp_baseline'},
      install_requires=['torch','selectivesearch'],
      version='0.0.1')

