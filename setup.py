
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ColemanCoalitions",
    version="0.0.1",
    author="Bjørn Erik Juel",
    author_email="bjorneju@gmail.com",
    description="A sef of functions implementing the concepts introduced by Coleman in 1973, and an extension to allow for the existence of coalitions within the collectve.",
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
