# Wizardry School Admissions Prediction

A supervised machine learning project that predicts which aspiring students are admitted to a prestigious **wizardry school**.  
The project applies a complete ML pipeline: data preprocessing, feature engineering, feature selection, model training & tuning, and evaluation on an **imbalanced dataset**.

Final model: **Decision Tree Classifier**, selected for its interpretability and balanced performance across accuracy, F1, precision, and recall.

---

## 🧰 Tech Stack

- **Python 3.10+**
- **Jupyter Notebook**
- **scikit-learn** – model building, preprocessing, feature selection, cross-validation
- **XGBoost** – gradient boosting model
- **imbalanced-learn** – handling dataset imbalance
- **NumPy / Pandas** – data manipulation
- **Matplotlib / Seaborn** – visualization

---

## 📁 Repository Structure

~~~text
.
├─ data/
│  └─ train.csv
│  └─ test.csv
├─ notebooks/
│  └─ ML1_Group08_Notebook.ipynb   # main notebook with pipeline
├─ report/
│  └─ ML1_Group08_Report.pdf       # detailed report
├─ requirements.txt
├─ model_metrics_xslx
├─ README.md
└─ LICENSE
~~~

---

## 🔎 Methodology

1. **Data Preparation**  
   - Missing value imputation (KNNImputer).  
   - Removal of low-value features (>50% missing).  
   - Data type optimization.  

2. **Exploration & Feature Engineering**  
   - Outlier detection & visualization.  
   - New features: *Experience Level Category*, *Financial Background Category*.  

3. **Feature Selection**  
   - Filter methods (variance, correlation, chi-square).  
   - Wrapper methods (RFE, Sequential Feature Selection).  
   - Embedded methods (Lasso, Decision Tree feature importances).  
   - Majority voting for final predictor set.  

4. **Modeling**  
   - Algorithms tested: Decision Tree, Extra Trees, Random Forest, Gradient Boosting, XGBoost, SVM, MLP.  
   - Hyperparameter tuning with GridSearchCV.  
   - Evaluation with K-Fold and Repeated K-Fold CV.  

5. **Evaluation Metrics**  
   - F1 Score (main metric for imbalanced data).  
   - Precision, Recall, Balanced Accuracy.  
   - Confusion Matrix.  

---

## 📊 Results

- **Best Model:** Decision Tree Classifier  
  - Tuned hyperparameters: `criterion="gini"`, `max_depth=5`, `max_leaf_nodes=7`, `min_samples_split=2`.  
  - Validation F1 Score: ~0.70  
  - Test F1 Score (Kaggle subset): 0.8888  
- Ensemble models like XGBoost and Gradient Boosting showed strong results, but the Decision Tree was chosen for interpretability.  
- Handling imbalance remains a future direction (oversampling, undersampling).  

---

## 🚀 How to Run

1. Clone this repository:  
   ~~~bash
   git clone https://github.com/<your-username>/wizardry-school-admissions-ml.git
   cd wizardry-school-admissions-ml
   ~~~

2. Install dependencies:  
   ~~~bash
   pip install -r requirements.txt
   ~~~

3. Open the notebook:  
   ~~~bash
   jupyter notebook notebooks/ML1_Group08_Notebook.ipynb
   ~~~

---

## ✅ Requirements

See [`requirements.txt`](./requirements.txt).

---

## 📄 License

This project is under MIT License

