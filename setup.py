from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="stateful_gaussian_splatting",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python liquid neural network project for stateful Gaussian splatting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pompoi/stateful_gaussian_splatting",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
)