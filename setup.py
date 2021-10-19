import warnings

try:
    import torch

    print(f"pytorch installation found with version {torch.__version__}")
except ImportError:
    warnings.warn(
        "Please install pytorch with gpu support using instructions at https://pytorch.org/"
    )
    raise

if not torch.cuda.is_available():
    warnings.warn(
        "Pytorch not found with gpu support, it is recommended to reinstall with gpu support"
    )

import shlex
import setuptools
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read()


class PostInstallCommand(install):
    def run(self):
        try:
            check_call(shlex.split("pre-commit install"))
        except Exception:
            print("\n\n\nUnable to run 'pre-commit install'\n\n\n")
        install.run(self)


setup(
    name="vast",
    version="0.0.1",
    author="Akshay Raj Dhamija",
    author_email="akshay.raj.dhamija@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    packages=setuptools.find_packages(),
    url="https://github.com/akshay-raj-dhamija/vast",
    cmdclass={
        "install": PostInstallCommand,
    },
)
