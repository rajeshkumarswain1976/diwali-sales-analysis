# Diwali Sales Analysis & Customer Prediction

Exploratory data analysis and machine learning on Diwali festival sales data.
Uncovers buying patterns across gender, age, state, occupation, and product category,
then builds a predictive model to identify high-spending customers.

---

### What it does

- Cleans and preprocesses real retail sales data — handles nulls, encodes
  categories, normalises numerical features
- Performs exploratory data analysis across 7 dimensions: gender, age group,
  state, marital status, occupation, product category, and product ID
- Engineers a binary target variable — HighSpender — based on 75th percentile
  of purchase amount
- Applies SMOTE to fix class imbalance before model training
- Trains and compares Logistic Regression and Random Forest classifiers
- Evaluates with accuracy, confusion matrix, and classification report

---

### Dataset

Diwali Sales Data — upload `Diwali_Sales_Data.csv` to your Google Drive
and update the file path in the notebook before running.

| Column | Description |
|---|---|
| Gender | Customer gender |
| Age Group | Age bracket |
| State | Indian state |
| Marital_Status | Married or single |
| Occupation | Customer occupation |
| Product_Category | Category of product purchased |
| Orders | Number of orders |
| Amount | Purchase amount (INR) |

---

### Key Insights from EDA

- Female customers have higher total purchasing power than male customers
- The 26–35 age group, particularly females, drives the highest sales
- Uttar Pradesh, Maharashtra, and Karnataka lead in both orders and sales amount
- Married women are the primary buyers during Diwali
- IT, Healthcare, and Aviation sector customers spend the most
- Food, Clothing, and Electronics are the top product categories

---

### ML Pipeline
Raw Data
→ Drop irrelevant columns + null rows
→ MinMaxScaler on Age, Orders, Amount
→ LabelEncoder on all categorical columns
→ Engineer HighSpender target (75th percentile threshold)
→ Train/Test split (80/20, stratified)
→ SMOTE on training set (fix class imbalance)
→ StandardScaler for Logistic Regression
→ Train Logistic Regression + Random Forest
→ Evaluate with accuracy + classification report

---

### Results

| Model | Accuracy |
|---|---|
| Logistic Regression | ~76% |
| **Random Forest** | **~82%** |

Random Forest outperforms Logistic Regression and can be used
to identify high-spending customers for targeted Diwali marketing campaigns.

---

### Stack

| | |
|---|---|
| **Data** | Pandas · NumPy |
| **Visualization** | Matplotlib · Seaborn |
| **ML** | scikit-learn · imbalanced-learn (SMOTE) |
| **Environment** | Google Colab |
| **Language** | Python |

---

### Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

# Upload Diwali_Sales_Data.csv to Google Drive
# Update file path in notebook cell 2
# Run all cells
```

---

### Images Folder

Run all cells and save these plots, then upload to `images/` folder:

| File | Plot |
|---|---|
| `gender_sales.png` | Total purchase amount by gender |
| `age_gender_sales.png` | Purchase amount by age group and gender |
| `state_orders.png` | Top 5 states by total orders |
| `state_amount.png` | Top 5 states by total sales amount |
| `marital_gender_sales.png` | Purchase by marital status and gender |
| `occupation_sales.png` | Top 5 occupations by purchase amount |
| `product_category_sales.png` | Top 5 product categories by sales |
| `top_products.png` | Top 10 most sold products by orders |
