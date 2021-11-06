import warnings

import shlex
import setuptools
from setuptools import setup
from setuptools.command.develop import develop
from subprocess import check_call

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read()
with open("dev-requirements.txt", "r", encoding="utf-8") as fh:
    dev_requirements = fh.read()


class PostDevelopCommand(develop):
    def run(self):
        try:
            check_call(shlex.split("pre-commit install"))
        except Exception as err:
            print(
                f"\n\n\n"
                f"Unable to run 'pre-commit install'\n"
                f"Ignore this message if you do not intend to make pull requests\n"
                f"{err}\n\n\n"
            )
        develop.run(self)


setup(
    name="vast",
    version="0.0.1",
    author="Akshay Raj Dhamija",
    author_email="akshay.raj.dhamija@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    extras_require={"dev": dev_requirements},
    packages=setuptools.find_packages(),
    url="https://github.com/Vastlab/vast",
    cmdclass={"develop": PostDevelopCommand},
)
