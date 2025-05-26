import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load training data
df = pd.read_csv('../datasets/train.csv')

# Extract features from datetime
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day
df['weekday'] = df['datetime'].dt.weekday
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year

# Drop columns that leak info or aren't needed
df = df.drop(['datetime', 'casual', 'registered'], axis=1)

# Split features and target
x = df.drop('count', axis=1)
y = df['count']

# Train linear regression model
model = LinearRegression()
model.fit(x, y)

# Load and process test data
test_df = pd.read_csv('../datasets/test.csv')
print(test_df.head())
test_df['datetime'] = pd.to_datetime(test_df['datetime'])
test_df['hour'] = test_df['datetime'].dt.hour
test_df['day'] = test_df['datetime'].dt.day
test_df['weekday'] = test_df['datetime'].dt.weekday
test_df['month'] = test_df['datetime'].dt.month
test_df['year'] = test_df['datetime'].dt.year

# Keep datetime separately for submission
datetime_col = test_df['datetime']

# Drop datetime for prediction
test_x = test_df.drop(['datetime'], axis=1)

# Make predictions
y_pred = model.predict(test_x)

# Clip negative values (bike count can't be negative)
y_pred = np.clip(y_pred, 0, None)

# Prepare submission
submission = pd.DataFrame({
    'datetime': datetime_col,
    'count': y_pred
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print(" Submission file created: submission.csv")
