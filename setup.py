import setuptools
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read()

setup(
    name='vast',
    version='0.0.1',
    author='Akshay Raj Dhamija',
    author_email='akshay.raj.dhamija@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    packages=setuptools.find_packages(),
    url='https://github.com/akshay-raj-dhamija/vast'
)
