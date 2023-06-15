import sys
import subprocess
from setuptools import setup

PY_VER = sys.version[0]
subprocess.call(["pip{:} install -r requirements.txt".format(PY_VER)], shell=True)

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name="nla",
    version="1.0.0",
    description="Natural Language Augmentor(nla) is the ultimate real and fast way to augment textual data for use in "
    "any model training. It has all the means to create natural and random textual errors.",
    url="https://github.com/pharmeasy/nla",
    author="Nikhil Kothari",
    author_email="nikhil.kothari@pharmeasy.in",
    license="Apache License 2.0",
    zip_safe=False,
    setup_requires=[],
    install_requires=[],
    packages=["nla"],
    package_data={"nla": ["nla/models/homophones/*"]},
    include_package_data=True,
    classifiers=["Programming Language :: Python :: 3"],
    long_description=long_description,
)
