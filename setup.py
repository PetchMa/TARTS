"""Setup configuration for Neural Active Optics System (NeuralAOS) package.

This setup script configures the installation of the NeuralAOS package, which implements
a deep learning-based active optics system for the LSST telescope. The package provides
neural network models for processing out-of-focus donut images to predict wavefront
aberrations and enable real-time telescope optics correction.
"""
from setuptools import setup, find_packages

setup(
    name="TARTS",
    version="0.0.0a1",
    packages=find_packages(),
    install_requires=[],
    author="Peter Xiangyuan Ma",
    author_email="peter_ma@berkeley.edu",
    description="I like to eat Tarts",
    license="MIT",
)
