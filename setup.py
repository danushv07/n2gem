import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="n2gem",
    version="0.0.1",
    author="Peter Steinbach, Danush Kumar V",
    author_email="p.steinbach@hzdr.de",
    description="nearest-neighbor generative model evaluation metrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.hzdr.de/haicu/internal-projects/danush_n2gem",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=['numpy','torch>=1.8.1',
                      'faiss-gpu==1.7.1'],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Bug Tracker": "https://gitlab.hzdr.de/haicu/internal-projects/danush_n2gem/-/issues",
    }
)
