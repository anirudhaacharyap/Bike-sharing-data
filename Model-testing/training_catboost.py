import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

# Load train data
train = pd.read_csv('../datasets/train.csv')
train['datetime'] = pd.to_datetime(train['datetime'])

# Feature engineering
train['hour'] = train['datetime'].dt.hour
train['day'] = train['datetime'].dt.day
train['weekday'] = train['datetime'].dt.weekday
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month

# Drop unwanted columns
train = train.drop(['datetime', 'casual', 'registered', 'month'], axis=1)

# Log-transform target
X = train.drop('count', axis=1)
y = np.log1p(train['count'])

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    verbose=100,
    random_seed=42
)

model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)

# Predict and calculate RMSLE
y_pred_val = model.predict(X_val)
rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(y_pred_val)))
print(f"Validation RMSLE: {rmsle:.5f}")

# Preprocess test data
test = pd.read_csv('../datasets/test.csv')
test_datetime = test['datetime']
test['datetime'] = pd.to_datetime(test['datetime'])

test['hour'] = test['datetime'].dt.hour
test['day'] = test['datetime'].dt.day
test['weekday'] = test['datetime'].dt.weekday
test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month

test = test.drop(['datetime', 'month'], axis=1)

# Predict
test_preds = model.predict(test)
test_preds = np.expm1(test_preds)
test_preds = np.clip(test_preds, 0, None)

# Create submission
submission = pd.DataFrame({
    'datetime': test_datetime,
    'count': test_preds
})

submission.to_csv('submission_catboost.csv', index=False)
print("Submission saved as 'submission_catboost.csv'")

# Plot predictions
plt.figure(figsize=(14, 5))
sns.lineplot(data=submission, x='datetime', y='count')
plt.title('CatBoost Predictions Over Time')
plt.xlabel('Datetime')
plt.ylabel('Predicted Count')
plt.tight_layout()
plt.show()
