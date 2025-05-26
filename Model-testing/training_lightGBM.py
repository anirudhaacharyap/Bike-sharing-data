import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error

# Load and preprocess train data
train = pd.read_csv('../datasets/train.csv')
train['datetime'] = pd.to_datetime(train['datetime'])
train['hour'] = train['datetime'].dt.hour
train['day'] = train['datetime'].dt.day
train['weekday'] = train['datetime'].dt.weekday
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month

# Drop unwanted columns
train = train.drop(['datetime', 'casual', 'registered', 'month'], axis=1)

# Features and target
X = train.drop('count', axis=1)
y = np.log1p(train['count'])  # Log transform target

# Train basic LightGBM model
model = LGBMRegressor(random_state=42)
model.fit(X, y)

# Preprocess test data
test = pd.read_csv('../datasets/test.csv')
test_datetime = test['datetime']
test['datetime'] = pd.to_datetime(test['datetime'])
test['hour'] = test['datetime'].dt.hour
test['day'] = test['datetime'].dt.day
test['weekday'] = test['datetime'].dt.weekday
test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month

# Drop month and datetime
test = test.drop(['datetime', 'month'], axis=1)

# Predict and inverse log transform
preds = model.predict(test)
preds = np.expm1(preds)  # Inverse of log1p
preds = np.clip(preds, 0, None)  # Avoid negatives

# Create submission
submission = pd.DataFrame({
    'datetime': test_datetime,
    'count': preds
})
submission.to_csv('submission_lgbm.csv', index=False)
print("Submission saved as 'submission_lgbm.csv'")
print(submission.head())

# Plot predictions
plt.figure(figsize=(14, 5))
sns.lineplot(data=submission, x='datetime', y='count')
plt.title('LightGBM Predictions Over Time')
plt.xlabel('Datetime')
plt.ylabel('Predicted Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
