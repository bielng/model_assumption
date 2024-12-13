# 📊 Linear Regression Model Assumptions Check 🌟

Welcome to the project documentation for the **Linear Regression Model Assumptions Check**. This project focuses on analyzing 💻 computer pricing data, building a regression model, and validating assumptions to ensure accurate and reliable predictions. 📈📉

---

## 🛠 Project Workflow 🔄

### 1. **Exploratory Data Analysis (EDA) 🔍**
- Visualized relationships between variables using:
  - 📊 Pair plots for pairwise relationships.
  - 📈 Bar plots to explore price variations across RAM categories.

```python
sns.pairplot(computers, corner=True)
computers.groupby('ram').agg({'price': 'mean'}).plot.bar()
```

### 2. **Model Specification 🧮**
- Target variable: `price` 💰
- Predictors: Features like `ram`, `speed`, `screen`, `ads`, and `trend`. 📋

```python
cols = ["ram", "speed", "hd", "screen", "ads", "trend"]
X = sm.add_constant(computers[cols])
y = computers["price"]
model = sm.OLS(y, X).fit()
```

### 3. **Assumptions Assessment ✅**

#### **Linearity 📏**
- Detected non-linearity.
- Added polynomial terms for improved fit. ✨

```python
computers = computers.assign(
    ram2 = computers['ram'] ** 2,
    hd2 = computers['hd'] ** 2,
    trend2 = computers['trend'] ** 2
)
```

#### **Independence of Errors 🎯**
- Checked residual plots for patterns.

#### **Normality of Errors 🎭**
- Log-transformed the target variable to address skewness.

```python
y = np.log(computers['price'])
```

#### **Multicollinearity 🔄**
- Checked Variance Inflation Factors (VIF) and dropped highly correlated variables.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
variables = sm.OLS(y, X).exog
pd.Series(
    [vif(variables, i) for i in range(variables.shape[1])],
    index=X.columns
)
```

#### **Equal Variance (Homoscedasticity) ⚖**
- Reviewed residual plots for consistent spread.

#### **Influential Points 📌**
- Identified high Cook’s Distance values and managed undue influence.

```python
influence = model.get_influence()
inf_summary_df = influence.summary_frame()
inf_summary_df['cooks_d'].sort_values(ascending=False).head()
```

### 4. **Model Refinement 🔧**
- Added transformed variables (`ram²`, `hd²`, `trend²`) and binary flags for categorical features (`premium`, `multi`, `cd`).
- Re-ran the regression model to ensure improved performance.

---

## 📈 Results 🎉
- **Improved Fit**: Incorporating transformed variables significantly enhanced the model’s predictive accuracy. 🥇
- **Assumptions Validated**:
  - Linearity ✅
  - Independence of Errors ✅
  - Normality of Errors ✅
  - Multicollinearity ✅
  - Equal Variance ✅
- **Robust Model**: Well-behaved residuals confirmed via diagnostic plots. 📊

```python
residual_analysis_plots(model)
```

---

## 🗒 Key Takeaways 💡
- **Feature Engineering**: Transformations like polynomial terms and log transformations can address violations of assumptions.
- **Diagnostics**: Tools like residual plots and Q-Q plots are invaluable for assumption validation.
- **Iterative Approach**: Re-specifying the model improves reliability and interpretability.

---

## 💻 Tools Used 🛠️
- **Languages & Libraries**: Python, Pandas, Statsmodels, Seaborn, Matplotlib, Scipy
- **Techniques**: EDA, Linear Regression, Assumption Diagnostics, Feature Engineering

---

## 🌟 Conclusion 🏁
This project highlights the importance of assumption validation in regression modeling to ensure robust and interpretable results. The process of identifying, addressing, and validating model assumptions is essential for data-driven decision-making. 📊📉

---

## 📬 Connect 🌐
If you have any questions or would like to collaborate, feel free to reach out! ✉️
