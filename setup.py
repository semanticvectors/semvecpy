import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="semvecpy",
    version="0.1.4",
    author="Semantic Vectors Authors",
    author_email="semanticvectors@googlegroups.com",
    description="Semantic Vectors work in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/semanticvector/semvecpy",
    packages=setuptools.find_packages(),
    python_requires='>=3',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "bitarray",
        "numpy",
    ],
)
