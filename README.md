# HousePricePrediction
# 🏡 House Price Prediction – Machine Learning Project  

A supervised machine learning project to predict house sale prices using Python, Pandas, scikit-learn, and XGBoost.  
This project is inspired by the Kaggle Ames Housing dataset.

---

## 📂 Project Overview  
This project applies data cleaning, feature engineering, and model training to predict house sale prices.  
Three different models are trained and evaluated — **Linear Regression, Random Forest, and XGBoost** — achieving **R² > 0.90** on validation data.

---

## ⚙️ Features  

- **Exploratory Data Analysis (EDA):** Histograms of price distribution & feature correlation heatmap.  
- **Data Cleaning:** Outlier removal, missing value imputation (median), encoding categorical variables.  
- **Feature Engineering:** Created new features such as total square footage and house age.  
- **Model Training:** Compared Linear Regression, Random Forest, and XGBoost.  
- **Evaluation:** R² score on validation data and 5-fold cross-validation.  
- **Visualization:** Actual vs. Predicted scatter plot for best-performing model.  

---

## 🛠️ Tech Stack  

- **Languages:** Python  
- **Libraries:** Pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn  

---

## 📊 Results  

| Model             | Validation R² | Mean CV R² |
|-------------------|--------------:|-----------:|
| Linear Regression | 0.8777        | 0.8723     |
| Random Forest     | 0.8939        | 0.8913     |
| XGBoost           | **0.9011**    | **0.9047** |

---

## 📸 Visualizations  

- `saleprice_distribution.png` – Distribution of house prices  
- `correlation_heatmap.png` – Correlation of features with SalePrice  
- `actual_vs_predicted.png` – Actual vs Predicted prices scatter plot  

---

## 🚀 How to Run  

1. Clone this repository:
   ```bash
   git clone https://github.com/YourUserName/HousePricePrediction.git
