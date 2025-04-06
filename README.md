
# 🧠 Obesity Risk Classification - Kaggle Playground Series 2024

This project was developed as part of a team submission to the **Kaggle Playground Series 2024**, focused on **multi-class prediction of obesity risk** based on a variety of lifestyle, demographic, and health-related features.

Our final model achieved an accuracy of **90%** on the public leaderboard — showcasing the effectiveness of our preprocessing pipeline and model selection.

---

## 🚀 Project Overview

The goal of this project is to classify individuals into one of several obesity risk categories using supervised machine learning. The final solution combines powerful ensemble learning techniques to deliver high prediction performance.

This predictive system can be leveraged in public health initiatives, wellness platforms, or research tools.

---

## 📊 Dataset

The dataset was provided as part of the Kaggle competition and includes anonymized data representing user habits, physical measures, and lifestyle choices. Key features include:

- **Age**, **Gender**
- **Physical activity level**
- **Consumption of high-calorie foods**
- **Transportation methods**
- **Water intake**, **Alcohol consumption**, etc.

The target variable is **NObesity**, a categorical feature with multiple risk classes:
- Insufficient Weight
- Normal Weight
- Overweight Level I
- Overweight Level II
- Obesity Type I
- Obesity Type II
- Obesity Type III

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **Jupyter Notebook**
- **NumPy, Pandas** – data manipulation
- **Matplotlib, Seaborn** – data visualization
- **Scikit-learn** – preprocessing, modeling, and evaluation
- **XGBoost** – gradient boosting classifier
- **Ensemble Techniques** – including Bagging and Boosting

---

## 📈 Methodology

1. **Exploratory Data Analysis (EDA)**  
   Visualized distributions and feature-target relationships using histograms, boxplots, and correlation heatmaps.

2. **Data Preprocessing**
   - Encoding categorical variables
   - Normalizing numerical features
   - Feature selection and importance analysis

3. **Model Development**
   - **Baseline models:** Logistic Regression, Decision Tree
   - **Ensemble methods:**
     - **Bagging** using Random Forest
     - **Boosting** using Gradient Boosting and **XGBoost**
   - Hyperparameter tuning and model comparison

4. **Model Evaluation**
   - Accuracy
   - Precision, Recall, F1-Score
   - Confusion Matrix
   - Cross-validation to assess generalization

5. **Submission to Kaggle**
   - Prepared predictions according to submission format
   - Achieved **90% accuracy** on the public leaderboard

---

## 💡 Ensemble Learning Techniques Used

| Method     | Description                                 |
|------------|---------------------------------------------|
| **Bagging** | Random Forests with multiple bootstrapped trees |
| **Boosting**| Gradient Boosting & XGBoost to reduce bias and variance |
| **Voting**  | Hard/soft voting used for combining multiple classifiers |

Ensemble learning greatly improved our prediction stability and accuracy.

---

## 🧪 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/MennatullahTarek/Multi-Class-Prediction-of-Obesity-Risk.git
   cd Multi-Class-Prediction-of-Obesity-Risk


2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the notebook:
   ```bash
   jupyter notebook classification_project_group_6.ipynb
   ```

> ⚠️ Ensure you have downloaded the competition dataset from Kaggle and placed it in the working directory.

---

## 🏆 Competition Info

- **Title:** [Kaggle Playground Series - Classification](https://www.kaggle.com/competitions/playground-series-s4e2/)
- **Year:** 2024
- **Type:** Multi-class classification
- **Metric:** Accuracy
- **Result:** ✅ 90% accuracy on public leaderboard

---

## 👥 Team Members

- **Mennatullah Tarek**  
- **Mariam Osama**
- **Yasmin Kadry**  
- **Aya Attia**

> We worked collaboratively on all stages including preprocessing, model building, and experimentation.

---

## 📄 License

This project is released for **educational and research purposes** only.  
You are welcome to reuse or adapt it with proper credit to the authors.

---

## 📬 Contact

For questions or collaborations, feel free to reach out:  
📧 **menatarek04@gmail.com**
📧 **mariamosama350@gmail.com**
📧 **yasminkadry6720@gmail.com**
📧 **ayaattia261@gmail.com**

