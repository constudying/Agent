#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name='agent',
    packages=[
        package for package in find_packages()
        if package.startswith("agent")
    ],
    version='0.0.0',
    description='',
    author='Chen Jihang',
    author_email='2517278658@qq.com',
    url='https://github.com/constudying/thesis.git',
    include_package_data=False,
    python_requires='==3.8.*',
)