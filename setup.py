from setuptools import setup, find_packages

setup(
    name="medoptix_ai_treatment_optimizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.95.0",
        "uvicorn==0.24.0",
        "pydantic>=1.6.2,<2.0.0",
        "pandas>=2.2.3",
        "numpy>=1.26.2",
        "scikit-learn>=1.3.2",
        "joblib>=1.3.2",
        "xgboost==2.0.2",
        "python-dateutil>=2.8.2",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "jupyter>=1.0.0",
        "notebook>=6.4.0",
        "statsmodels>=0.13.0",
        "pytest==8.4.0",
        "pytest-cov==4.1.0",
        "pytest-asyncio==0.23.5",
        "httpx==0.24.1"
    ],
    python_requires=">=3.8",
) 