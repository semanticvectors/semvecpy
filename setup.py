import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="semvecpy",
    version="0.1",
    author="Several contributors including Dominic Widdows",
    author_email="dwiddows@gmail.com",
    description="Semantic Vectors work in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/semanticvector/semvecpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3-Clause",
        "Operating System :: OS Independent",
    ],
)