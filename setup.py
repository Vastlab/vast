import setuptools
from setuptools import setup
from setuptools.command.develop import develop

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read()
class PostInstallCommand(develop):
    def run(self):
        try:
            check_call(shlex.split("pre-commit install"))
        except Exception as e:
            logger.warning("Unable to run 'pre-commit install'")
        develop.run(self)

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
