from setuptools import setup, find_packages

setup(
    name="medoptix_ai_treatment_optimizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib",
        "pytest",
        "pytest-cov",
        "pytest-asyncio"
    ],
    python_requires=">=3.12",
) 