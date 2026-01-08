from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='hashformers',
    version='2.2.0',
    author='Ruan Chaves Rodrigues',
    author_email='ruanchave93@gmail.com',
    description='Word segmentation with transformers',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ruanchaves/hashformers",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    python_requires=">=3.8",
    install_requires=[
        "minicons",
        "twitter-text-python",
        "pandas"
    ],
    extras_require={
        "spacy": ["spacy>=3.0.0"]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="word-segmentation hashtag nlp transformers spacy",
)
