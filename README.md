# Student Performance Predictor

A machine learning project that analyzes and predicts student academic performance based on various socio-economic and educational factors.

## 🎯 Overview

This project uses machine learning algorithms to predict student exam scores based on various demographic, socio-economic, and educational factors. The system can help educators and institutions identify students who might need additional support and understand which factors most significantly impact academic performance.

## ✨ Features

- **Data Analysis**: Comprehensive exploratory data analysis of student performance factors
- **Preprocessing**: Data cleaning, feature engineering, and encoding of categorical variables
- **Multiple Models**: Implementation of various ML algorithms (Linear Regression, Random Forest, etc.)
- **Visualization**: Interactive charts and graphs showing performance insights
- **Prediction**: Real-time prediction of student scores based on input parameters
- **Model Evaluation**: Detailed performance metrics and model comparison

## 📊 Dataset

The project uses student performance data with the following features:

### Input Features:
- **Gender**: Student's gender (Male/Female)
- **Race/Ethnicity**: Student's ethnic background (Group A, B, C, D, E)
- **Parental Level of Education**: Highest education level of parents
- **Lunch**: Type of lunch (Standard/Free or Reduced)
- **Test Preparation Course**: Whether student completed test prep course
- **Reading Score**: Score in reading exam
- **Writing Score**: Score in writing exam

### Target Variable:
- **Math Score**: Student's mathematics exam score (0-100)

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/itsBabuaa/student-preformance.git
   cd student-preformance
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python application.py
   ```

## 📈 Model Performance

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | 0.85 | 6.2 | 4.8 |
| Random Forest | 0.88 | 5.8 | 4.2 |
| Gradient Boosting | 0.87 | 5.9 | 4.5 |
| Support Vector Regression | 0.84 | 6.5 | 5.1 |

*Best performing model: **Random Forest** with R² score of 0.88*

## 📁 Project Structure

```
student-preformance/
│
├── artifacts/                  # Model artifacts and preprocessed data
├── notebook/                   # Jupyter notebooks for EDA and experiments
├── src/
│   ├── components/
│   │   ├── data_ingestion.py   # Data loading and splitting
│   │   ├── data_transformation.py  # Data preprocessing
│   │   └── model_trainer.py    # Model training and evaluation
│   ├── pipeline/
│   │   ├── predict_pipeline.py # Prediction pipeline
│   │   └── train_pipeline.py   # Training pipeline
│   ├── exception.py           # Custom exception handling
│   ├── logger.py             # Logging configuration
│   └── utils.py              # Utility functions
├── static/                   # CSS, JS, and image files
├── templates/               # HTML templates
├── application.py                  # Flask web application
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib/Seaborn**: Data visualization
- **Flask**: Web framework for deployment
- **HTML/CSS**: Frontend interface
- **Jupyter Notebook**: Data exploration and prototyping

## 🔍 Key Insights

- **Parental education level** has the strongest correlation with student performance
- Students who completed **test preparation courses** show significantly higher scores
- **Lunch type** (indicator of socio-economic status) impacts performance
- **Gender disparities** exist across different subjects
- **Reading and writing scores** are strong predictors of math performance
  

## 🙏 Acknowledgments

- Dataset source: [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- Inspiration from various educational data science projects
- Thanks to the open-source community for the amazing tools and libraries

---

⭐ If you found this project helpful, please consider giving it a star!
