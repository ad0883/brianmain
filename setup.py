from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="brain_tumor_detection",
    version="1.0.0",
    author="",
    author_email="project@drdo.gov.in",
    description="Brain Tumor Detection System using Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/drdo/brain-tumor-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
        "tensorflow>=2.10.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "flask>=2.0.0",
        "flask-cors>=3.0.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "brain-tumor-train=src.train:train_model",
            "brain-tumor-predict=src.predict:predict_image",
            "brain-tumor-web=app.app:main",
        ],
    },
)
