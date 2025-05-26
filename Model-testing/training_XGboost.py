import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_log_error, make_scorer, mean_squared_log_error
from xgboost import XGBRegressor

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
train = train.drop(['datetime', 'casual', 'registered','month'], axis=1)

X = train.drop('count', axis=1)
y = np.log1p(train['count'])

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, np.maximum(0, y_pred)))
rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.3],
    'subsample': [0.8, 1.0]
}

# Train basic XGBoost model
model = XGBRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=rmsle_scorer,
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X, y)

best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
# Load and preprocess test data
test = pd.read_csv('../datasets/test.csv')
test_datetime = test['datetime']
test['datetime'] = pd.to_datetime(test['datetime'])
test['hour'] = test['datetime'].dt.hour
test['day'] = test['datetime'].dt.day
test['weekday'] = test['datetime'].dt.weekday
test['month'] = test['datetime'].dt.month
test['year'] = test['datetime'].dt.year

test = test.drop(['datetime', 'month'], axis=1)

# Predict
preds =best_model.predict(test)
preds = np.expm1(preds)
preds = np.clip(preds, 0, None)

# Submission
submission = pd.DataFrame({
    'datetime': test_datetime,
    'count': preds
})

# Line Plot
'''plt.figure(figsize=(14, 5))
sns.lineplot(data=submission, x='datetime', y='count')
plt.title('XGBoost Predictions Over Time')
plt.xlabel('Datetime')
plt.ylabel('Predicted Count')
plt.tight_layout()
plt.show()

# Scatter Plot
plt.figure(figsize=(14, 5))
sns.scatterplot(data=submission, x='datetime', y='count', s=10, color='orange')
plt.title('Scatter Plot: XGBoost Predicted Bike Rentals')
plt.xlabel('Datetime')
plt.ylabel('Predicted Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
'''
# Save
submission.to_csv('submission_xgboost_Gridsearch.csv', index=False)
print("Submission saved as 'submission_xgboost_basic.csv'")