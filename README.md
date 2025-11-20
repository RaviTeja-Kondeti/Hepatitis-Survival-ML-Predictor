# 🏥 Hepatitis Survival ML Predictor

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive machine learning project that predicts hepatitis patient survival outcomes using ensemble learning techniques, featuring advanced model stacking, hyperparameter optimization, and feature importance analysis.

## 🎯 Project Overview

This project implements a complete ML pipeline to predict survival outcomes for hepatitis patients by analyzing both categorical and numeric clinical features. The system leverages multiple classification algorithms and combines them through ensemble learning for improved prediction accuracy.

### Key Features

- 🤖 **Multiple ML Models**: Implementation of 4 distinct classifiers (SVM, Decision Tree, Random Forest, KNN)
- 📊 **Ensemble Learning**: Advanced stacking approach with MLPClassifier as meta-learner
- 🔧 **Hyperparameter Optimization**: Randomized search for optimal model configuration
- 📈 **Feature Importance Analysis**: Identification of top 5 predictive features
- 📉 **Comprehensive Evaluation**: Multi-metric performance assessment (accuracy, precision, recall, F1-score)

## 🏗️ Architecture

```
Data Input → Preprocessing → Model Training → Ensemble Stacking → Predictions
                ↓                  ↓                   ↓
           Encoding          Base Models        Meta-Learner
           Scaling           (4 types)          (MLP)
```

## 🚀 Models Implemented

1. **Support Vector Machine (LinearSVC)** - Linear classification with maximum margin
2. **Decision Tree Classifier** - Rule-based hierarchical decisions
3. **Random Forest Classifier** - Ensemble of decision trees with bagging
4. **K-Nearest Neighbors** - Instance-based learning algorithm
5. **Stacking Ensemble** - MLPClassifier meta-learner combining all base models

## 📊 Dataset

The project uses the Hepatitis dataset containing:
- Multiple clinical features (both categorical and numeric)
- Patient demographics
- Medical history indicators
- Survival outcome labels

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning algorithms and tools
- **Google Colab** - Development and experimentation environment

## 📈 Results & Performance

The project achieves strong predictive performance through:
- ✅ Baseline model comparison across multiple metrics
- ✅ Hyperparameter-tuned Random Forest optimization
- ✅ Enhanced predictions via model stacking
- ✅ Feature importance ranking for clinical insights

## 🔬 Methodology

1. **Data Preprocessing**
   - Label encoding for categorical variables
   - Feature scaling using StandardScaler
   - Train-test split for model validation

2. **Model Training & Evaluation**
   - Individual model training and assessment
   - Performance comparison using multiple metrics
   - Cross-validation for robust evaluation

3. **Hyperparameter Tuning**
   - Randomized search on Random Forest
   - Optimization of key parameters
   - Performance improvement analysis

4. **Feature Analysis**
   - Extraction of feature importance scores
   - Identification of top predictive features
   - Clinical interpretation of results

5. **Ensemble Stacking**
   - Base model prediction combination
   - MLPClassifier as meta-learner
   - Performance boost through ensemble

## 💡 Key Insights

- Feature importance analysis reveals the most critical clinical indicators for survival prediction
- Ensemble stacking consistently outperforms individual base models
- Hyperparameter tuning provides measurable improvements in Random Forest performance
- Model diversity in the ensemble contributes to robust predictions

## 🔍 Future Enhancements

- [ ] Integration of additional advanced models (XGBoost, LightGBM)
- [ ] Cross-validation with multiple folds for more robust evaluation
- [ ] Feature engineering for enhanced predictive power
- [ ] Model interpretability tools (SHAP, LIME)
- [ ] Web-based deployment for clinical use
- [ ] Real-time prediction API

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Ravi Teja Kondeti**

- GitHub: [@RaviTeja-Kondeti](https://github.com/RaviTeja-Kondeti)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Hepatitis dataset
- scikit-learn development team for excellent ML tools
- Open-source community for inspiration and resources

---

<div align="center">
  <strong>⭐ Star this repository if you find it helpful!</strong>
</div>
