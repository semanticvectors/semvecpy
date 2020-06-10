import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as fh:
    requirements = fh.read().splitlines()
    
setuptools.setup(
    name="semvecpy",
    version="0.1.11",
    author="Semantic Vectors Authors",
    author_email="semanticvectors@googlegroups.com",
    description="Semantic Vectors work in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/semanticvectors/semvecpy",
    packages=setuptools.find_packages(),
    python_requires=">=3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements
)
