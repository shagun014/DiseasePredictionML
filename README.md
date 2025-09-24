# DiseasePredictionML

# Disease Prediction ML System 🏥

A comprehensive machine learning system for predicting diseases (Heart Disease and Diabetes) using multiple classification algorithms with an interactive user interface.

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.6+-green.svg)](https://xgboost.readthedocs.io)

## 🌟 Features

- **Multi-Disease Prediction**: Heart Disease and Diabetes prediction models
- **Multiple ML Algorithms**: SVM, Logistic Regression, Random Forest, XGBoost
- **Interactive Interface**: User-friendly command-line interface for data input
- **Comprehensive Evaluation**: Cross-validation, ROC curves, confusion matrices
- **Real-time Predictions**: Instant risk assessment with probability scores
- **Data Visualization**: Performance comparison plots and feature importance analysis

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.7+
pip (Python package installer)
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/shagun014/disease-prediction-ml.git
cd disease-prediction-ml
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the program:
```bash
python disease_prediction.py
```

## 📋 Usage

### Interactive Mode
Run the program and follow the prompts to enter patient data:

```bash
python disease_prediction.py
```

Choose between:
1. **Interactive Mode** - Enter real patient data
2. **Quick Demo Mode** - See sample predictions

### Example Input (Heart Disease)
```
Enter age (years): 58
Enter sex (M/F or 1/0): M
Enter chest pain type (0-3): 2
Enter resting blood pressure (mmHg): 145
Enter cholesterol level (mg/dl): 280
...
```

### Example Output
```
============================================================
PREDICTION RESULTS
============================================================
Disease: Heart Disease
Prediction: POSITIVE
Probability: 73.2%
Risk Level: High

⚠️ HIGH RISK: Strong indication of heart disease. 
   Immediate medical consultation recommended.
```

## 🧠 Machine Learning Models

### Algorithms Used
1. **Support Vector Machine (SVM)** - Effective for high-dimensional data
2. **Logistic Regression** - Interpretable linear model
3. **Random Forest** - Ensemble method resistant to overfitting
4. **XGBoost** - Gradient boosting for high performance

### Performance Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Cross-validation scores

## 📊 Datasets

### Heart Disease Features
- Age, Sex, Chest Pain Type
- Resting Blood Pressure, Cholesterol
- Fasting Blood Sugar, Resting ECG
- Maximum Heart Rate, Exercise Angina
- ST Depression, ST Slope

### Diabetes Features
- Pregnancies, Glucose Level
- Blood Pressure, Skin Thickness
- Insulin Level, BMI
- Diabetes Pedigree Function, Age

## 📈 Model Performance

The system provides comprehensive model evaluation:

- **Cross-validation** for robust performance estimation
- **ROC curves** for all models
- **Confusion matrices** for detailed analysis
- **Feature importance** visualization
- **Model comparison** charts

## 🔬 Google Colab Integration

To run in Google Colab:

1. Upload the notebook to Google Drive
2. Open with Google Colab
3. Install requirements:
```python
!pip install xgboost
```
4. Run all cells

## 📁 Project Structure

```
disease-prediction-ml/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── disease_prediction.py     # Main Python script
├── notebooks/
│   └── disease_prediction.ipynb  # Jupyter notebook version
├── data/
│   └── sample_data/         # Sample datasets
└── images/
    └── screenshots/         # Result visualizations
```

## ⚙️ Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
xgboost>=1.6.0
```

## 🎯 Use Cases

- **Healthcare Professionals**: Risk assessment tool
- **Medical Research**: Disease pattern analysis
- **Educational**: ML in healthcare demonstration
- **Personal Health**: Self-assessment (not a substitute for medical advice)

## 📊 Results Interpretation

### Risk Levels
- **High Risk (>70%)**: Immediate medical consultation recommended
- **Medium Risk (30-70%)**: Consider medical evaluation
- **Low Risk (<30%)**: Continue regular health monitoring

### Model Confidence
The system shows predictions from all models to provide confidence intervals and consensus-based decisions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚠️ Important Disclaimer

**This system is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.**

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for dataset inspiration
- Scikit-learn community for excellent documentation
- XGBoost developers for the powerful gradient boosting library

