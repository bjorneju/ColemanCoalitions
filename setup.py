
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="coleman_coalitions",
    version="0.1.0",
    author="Bjørn Erik Juel",
    author_email="bjorneju@gmail.com",
    description="Python implementation of Coleman's (1973) mathematical framework for collective action and coalition analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bjorneju/ColemanCoalitions/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNUv3 License",
        "Operating System :: OS Independent",
    ],
)
