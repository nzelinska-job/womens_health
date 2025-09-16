from setuptools import setup, find_packages

setup(
    name="womens-health",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "tensorflow>=2.10.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Women's health and menstrual cycle prediction",
    python_requires=">=3.8",
)
