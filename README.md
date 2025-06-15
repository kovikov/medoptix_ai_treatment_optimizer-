# MedOptix AI Treatment Optimizer

A machine learning-powered healthcare application that predicts patient treatment adherence and dropout risks using XGBoost, FastAPI, and Streamlit.

## 🏥 Overview

MedOptix AI Treatment Optimizer is an intelligent system designed to help healthcare providers predict and prevent treatment non-adherence. By analyzing patient data and treatment patterns, the system provides:

- Dropout risk predictions
- Treatment adherence forecasts
- Personalized intervention recommendations
- Real-time monitoring and alerts

## ✨ Features

- **Dropout Risk Prediction**: Identifies patients at risk of discontinuing treatment
- **Adherence Forecasting**: Predicts future treatment adherence levels
- **Real-time Monitoring**: Tracks patient progress and engagement
- **Interactive Dashboard**: Visualizes patient data and predictions
- **API Integration**: Easy integration with existing healthcare systems
- **Data Privacy**: Secure handling of patient information

## 🛠️ Technology Stack

- **Machine Learning**: XGBoost, scikit-learn
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Testing**: pytest, pytest-cov
- **Deployment**: Docker, GitHub Actions

## 📋 Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/medoptix_ai_treatment_optimizer.git
cd medoptix_ai_treatment_optimizer
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## 💻 Usage

### Running the API Server

```bash
uvicorn medoptix_ai_treatment_optimizer.app:app --reload
```

The API will be available at `http://localhost:8000`

### Running the Streamlit Dashboard

```bash
streamlit run medoptix_ai_treatment_optimizer/dashboard.py
```

The dashboard will be available at `http://localhost:8501`

### API Endpoints

- `POST /predict_dropout`: Predict patient dropout risk
- `POST /forecast_adherence`: Forecast treatment adherence
- `GET /docs`: Interactive API documentation

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run tests with coverage:
```bash
pytest tests/ --cov=. --cov-report=xml --cov-report=term-missing
```

## 📊 Project Structure

```
medoptix_ai_treatment_optimizer/
├── data/                  # Data directory
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data files
├── models/               # Trained model files
├── notebooks/            # Jupyter notebooks
├── tests/                # Test files
├── medoptix_ai_treatment_optimizer/
│   ├── app.py           # FastAPI application
│   ├── dashboard.py     # Streamlit dashboard
│   ├── train_models.py  # Model training
│   └── utils.py         # Utility functions
├── requirements.txt      # Project dependencies
└── setup.py             # Package setup file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Healthcare professionals who provided domain expertise
- Open-source community for the amazing tools and libraries
- Contributors and maintainers of the project

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers.