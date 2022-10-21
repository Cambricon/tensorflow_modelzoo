import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="record_time",
  version="0.0.1",
  author="Cambricon Corporation",
  description="A small timing scripts",
  long_description=long_description,
  long_description_content_type="text/markdown",
  packages=setuptools.find_packages(),
  classifiers=[
  "Programming Language :: Python :: 3",
  "Operating System :: OS Independent",
  ],
)