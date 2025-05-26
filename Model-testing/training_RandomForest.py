import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


# Load and preprocess training data
train = pd.read_csv('../datasets/train.csv')

# Extract date-time features
train['datetime'] = pd.to_datetime(train['datetime'])
train['hour'] = train['datetime'].dt.hour
train['day'] = train['datetime'].dt.day
train['weekday'] = train['datetime'].dt.weekday
train['month'] = train['datetime'].dt.month
train['year'] = train['datetime'].dt.year

# Drop unnecessary columns
train = train.drop(['datetime', 'casual', 'registered'], axis=1)

# Split into features and target
X = train.drop('count', axis=1)
y = train['count']


# Train the Random Forest model on full training data
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Load and preprocess test data
test = pd.read_csv('../datasets/test.csv')

# Store datetime separately for submission
test_datetime = test['datetime']

# Convert and extract time features
test['datetime'] = pd.to_datetime(test['datetime'])
test['hour'] = test['datetime'].dt.hour
test['day'] = test['datetime'].dt.day
test['weekday'] = test['datetime'].dt.weekday
test['month'] = test['datetime'].dt.month
test['year'] = test['datetime'].dt.year



# Drop extra columns to match training data
test = test.drop(['datetime'], axis=1)

# Predict
test_preds = model.predict(test)   # Fixed here: replaced X_test with test
test_preds = np.clip(test_preds, 0, None)  # Avoid negative values

# Create submission file
submission = pd.DataFrame({
    'datetime': test_datetime,
    'count': test_preds
})
x=submission.datetime
z=submission.count

submission.to_csv('submission_random_forest.csv', index=False)
print("Submission file saved as 'submission_random_forest.csv'")
