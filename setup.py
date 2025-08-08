"""Setup configuration for TARTS (Neural Active Optics System) package.

This setup script configures the installation of the TARTS package, which implements
a deep learning-based active optics system for the LSST telescope. The package provides
neural network models for processing out-of-focus donut images to predict wavefront
aberrations and enable real-time telescope optics correction.
"""
from setuptools import setup, find_packages

setup(
    name="tarts",
    version="0.0.0a3",
    packages=find_packages(),
    install_requires=[],
    author="Peter Xiangyuan Ma",
    author_email="peter_ma@berkeley.edu",
    description="Neural Active Optics System for LSST telescope",
    license="MIT",
)
