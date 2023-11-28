import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="llmdet",
    version="1.1.0",
    author="Kangxi Wu, Liang Pang",
    author_email="wukx0901@gmail.com",
    description="LLMDet: A Third Party Large Language Models Generated Text Detection Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cansee5/LLMDet",
    packages=setuptools.find_packages(),
    install_requires=[
            'nltk',
            'lightgbm',
            'numpy',
            'scikit-learn',
            'tqdm',
            'argparse',
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
