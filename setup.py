import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tsp",
    version="0.0.1",
    author="Maksym Shpakovych",
    author_email="maksym.shpakovych@inria.fr",
    description="Graph convolution network for TSP.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/shaxov/tsp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
