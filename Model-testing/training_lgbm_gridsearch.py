import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_log_error

# Load and preprocess training data
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
y = np.log1p(train['count'])

# Define RMSLE scorer
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, np.maximum(0, y_pred)))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [6, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_samples': [20, 30]
}

# Initialize model
lgbm = LGBMRegressor(random_state=42)

# Grid search
grid_search = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    scoring=rmsle_scorer,
    cv=3,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X, y)
best_model = grid_search.best_estimator_
print("Best parameters found:", grid_search.best_params_)

# Load and preprocess test data
test = pd.read_csv('../datasets/test.csv')
test_datetime = test['datetime']
test['datetime'] = pd.to_datetime(test['datetime'])
test['hour'] = test['datetime'].dt.hour
test['day'] = test['datetime'].dt.day
test['weekday'] = test['datetime'].dt.weekday
test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test = test.drop(['datetime', 'month'], axis=1)

# Predict and revert log1p
preds = best_model.predict(test)
preds = np.expm1(preds)
preds = np.clip(preds, 0, None)

# Create submission
submission = pd.DataFrame({
    'datetime': test_datetime,
    'count': preds
})
submission.to_csv('submission_lgbm_gridsearch.csv', index=False)
print("Submission saved as 'submission_lgbm_gridsearch.csv'")

# Plot results
plt.figure(figsize=(14, 5))
sns.lineplot(data=submission, x='datetime', y='count')
plt.title('LightGBM Predictions Over Time (GridSearchCV)')
plt.xlabel('Datetime')
plt.ylabel('Predicted Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
