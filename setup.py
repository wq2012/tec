"""Setup script for the Textual Echo Cancellation (tec) package."""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as rq:
  install_requires = rq.read().splitlines()

setuptools.setup(
    name="textual-echo-cancellation",
    version="0.1.2",
    author="Quan Wang",
    author_email="quanw@google.com",
    description="Textual Echo Cancellation (TEC) with Multi-Source Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wq2012/tec",
    packages=setuptools.find_packages(include=["tec", "tec.*"]),
    scripts=[
        "scripts/prepare_data.py",
        "scripts/train.py",
        "scripts/inference.py",
        "scripts/evaluate.py",
        "scripts/export_tflite.py",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
)
