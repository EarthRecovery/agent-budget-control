from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='robotouille',
    version='1.0',
    author='',
    author_email='',
    description='A challenging benchmark for testing LLM agent planning capabilities!',
    packages=find_packages(),
    install_requires=requirements,
)
