from pathlib import Path
from setuptools import setup, find_packages

VERSION = "1.0.0"
DESCRIPTION = (
    "Compute operational differences between two "
    "sequences/texts using the Levenshtein algorithm"
)

root = Path(__file__).parent
long_description = (root / "README.md").read_text()


setup(
    name="levenshtein-distance",
    author="syn-chromatic",
    author_email="synchromatic.github@gmail.com",
    url="https://github.com/syn-chromatic/levenshtein-distance",
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=["setuptools>=45.0"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.7",
        "Topic :: Utilities",
    ],
)
