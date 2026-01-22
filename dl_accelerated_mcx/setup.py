"""
Setup script for DL Accelerated MCX Simulation package
"""
from setuptools import setup, find_packages


def read_requirements():
    """Read requirements from requirements.txt."""
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="dl_accelerated_mcx",
    version="1.0.0",
    description="Deep Learning Accelerated Monte Carlo eXtreme Simulation for Transcranial Photobiomodulation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Shiyu Liu",
    author_email="syliu@example.com",
    url="https://github.com/syliu/dl_accelerated_mcx",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dl-mcx=scripts.cli:main",
            "dl-mcx-gui=scripts.gui:main",
            "dl-mcx-web=scripts.web:main",
        ],
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="monte carlo simulation, photobiomodulation, deep learning, medical imaging, nifti",
    project_urls={
        "Bug Reports": "https://github.com/syliu/dl_accelerated_mcx/issues",
        "Source": "https://github.com/syliu/dl_accelerated_mcx",
        "Documentation": "https://github.com/syliu/dl_accelerated_mcx/blob/main/README.md",
    },
)