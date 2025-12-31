from setuptools import setup, find_packages

setup(
    name="aibetting",
    version="0.1.0",
    description="AI-powered betting prediction system",
    author="Andy Cheng",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "aibetting=aibetting.cli:main",
        ],
    },
)
