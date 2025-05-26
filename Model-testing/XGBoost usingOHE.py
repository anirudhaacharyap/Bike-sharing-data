import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# --- Load and preprocess training data ---
train = pd.read_csv('../datasets/train.csv')
train['datetime'] = pd.to_datetime(train['datetime'])

# Extract features
train['hour'] = train['datetime'].dt.hour
train['day'] = train['datetime'].dt.day
train['weekday'] = train['datetime'].dt.weekday
train['month'] = train['datetime'].dt.month
train['year'] = train['datetime'].dt.year

# Save and remove target
y = np.log1p(train['count'])  # log1p for RMSLE
train = train.drop(['datetime', 'casual', 'registered', 'count'], axis=1)

# One-hot encode 'month' sparsely
encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=True)
month_encoded = encoder.fit_transform(train[['month']])
train.drop('month', axis=1, inplace=True)

# Combine encoded month with numeric features
X_numeric = sparse.csr_matrix(train.values)
X = sparse.hstack([X_numeric, month_encoded])

# --- Train basic XGBoost regressor ---
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    tree_method='auto'
)
model.fit(X, y)

# --- Preprocess test data ---
test = pd.read_csv('../datasets/test.csv')
test_datetime = test['datetime']
test['datetime'] = pd.to_datetime(test['datetime'])

test['hour'] = test['datetime'].dt.hour
test['day'] = test['datetime'].dt.day
test['weekday'] = test['datetime'].dt.weekday
test['month'] = test['datetime'].dt.month
test['year'] = test['datetime'].dt.year

test = test.drop(['datetime'], axis=1)

# Encode test month using same encoder
month_encoded_test = encoder.transform(test[['month']])
test.drop('month', axis=1, inplace=True)

X_test_numeric = sparse.csr_matrix(test.values)
X_test = sparse.hstack([X_test_numeric, month_encoded_test])

# --- Predict and prepare submission ---
test_preds = model.predict(X_test)
test_preds = np.expm1(test_preds)  # undo log1p
test_preds = np.clip(test_preds, 0, None)

submission = pd.DataFrame({
    'datetime': test_datetime,
    'count': test_preds
})

# --- Plots ---
plt.figure(figsize=(14, 5))
sns.lineplot(data=submission, x='datetime', y='count')
plt.title('Predicted Bike Rentals Over Time')
plt.xlabel('Datetime')
plt.ylabel('Predicted Count')
plt.tight_layout()
plt.show()

'''plt.figure(figsize=(14, 5))
sns.scatterplot(data=submission, x='datetime', y='count', s=10, color='green')
plt.title('Scatter Plot: Predicted Bike Rentals Over Time')
plt.xlabel('Datetime')
plt.ylabel('Predicted Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()'''

# --- Save submission ---
submission.to_csv('submission_xgboost_sparse_month.csv', index=False)
print("Submission saved as 'submission_xgboost_sparse_month.csv'")
