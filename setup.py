from setuptools import setup, find_packages

setup(
    name="green_score",
    version="0.0.12",
    author="Sophie Ostmeier, Jean-Benoit Delbrouck",
    license="MIT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
    packages=find_packages(),
    zip_safe=False,
)
