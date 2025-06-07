# setup.py

from setuptools import setup, find_packages
from pathlib import Path

# lies die README als Long Description ein (Markdown)
here = Path(__file__).parent
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="vitalnodes",
    version="0.1.0",
    description="Gravity- und Entropie-basierte Zentralitätsmetriken für NetworkX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Dein Name",
    author_email="deine.email@example.com",
    url="https://github.com/deinusername/vitalnodes",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "networkx>=2.5",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",      # zum Testen
            "flake8>=4.0",      # für Linting
        ],
    },
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "vitalnodes-cli = vitalnodes.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
