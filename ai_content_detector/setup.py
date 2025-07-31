"""Setup script for AI Content Detection System."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-content-detector",
    version="1.0.0",
    author="AI Content Detection Team",
    author_email="team@ai-content-detector.com",
    description="A hybrid AI-generated text and image detector for high-accuracy, low-cost, and production-scale deployment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/ai-content-detector",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.0.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-detector-api=api.main:main",
            "ai-detector-ui=ui.app:main",
            "ai-detector-eval=scripts.evaluate_models:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "*.md"],
    },
    keywords="ai detection, text detection, image detection, machine learning, deep learning",
    project_urls={
        "Bug Reports": "https://github.com/your-org/ai-content-detector/issues",
        "Source": "https://github.com/your-org/ai-content-detector",
        "Documentation": "https://ai-content-detector.readthedocs.io/",
    },
)