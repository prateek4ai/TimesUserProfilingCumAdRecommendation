"""Setup configuration for times-ctr-optimizer"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="times-ctr-optimizer",
    version="1.0.0",
    author="Prateek",
    author_email="prat.cann.170701@gmail.com",
    description="Enterprise CTR prediction system for Times Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/prateek4ai/TimesUserProfilingCumAdRecommendation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "polars>=0.19.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "black>=23.0", "flake8>=6.0"],
        "api": ["fastapi>=0.104.0", "uvicorn[standard]>=0.24.0"],
    },
    entry_points={
        "console_scripts": [
            "times-ctr=times_ctr_optimizer.cli:main",
        ],
    },
)
