import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Dhamija",
    version="0.0.1",
    author="Akshay Raj Dhamija",
    author_email="adhamija@vast.uccs.edu",
    description="A package for some common operations for everyone",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Vastlab/utile",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Original BSD License",
        "Operating System :: Linux",
    ],
    python_requires='>=3.8',
)
