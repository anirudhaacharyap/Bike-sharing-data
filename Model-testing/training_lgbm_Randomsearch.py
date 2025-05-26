import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_log_error
from scipy.stats import randint, uniform

# Load train data
train = pd.read_csv('../datasets/train.csv')
train['datetime'] = pd.to_datetime(train['datetime'])

# Feature engineering
train['hour'] = train['datetime'].dt.hour
train['day'] = train['datetime'].dt.day
train['weekday'] = train['datetime'].dt.weekday
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month

# Drop unused
train = train.drop(['datetime', 'casual', 'registered', 'month'], axis=1)

# Features & target
X = train.drop('count', axis=1)
y = np.log1p(train['count'])

# Define RMSLE scorer
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, np.maximum(0, y_pred)))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

# Hyperparameter space
param_dist = {
    'n_estimators': randint(100, 500),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(3, 12),
    'num_leaves': randint(20, 150),
    'min_child_samples': randint(5, 50),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1)
}

# Initialize model
lgbm = LGBMRegressor(random_state=42)

# Randomized Search
random_search = RandomizedSearchCV(
    estimator=lgbm,
    param_distributions=param_dist,
    n_iter=50,  # Increase this to 100+ for even deeper tuning
    scoring=rmsle_scorer,
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X, y)
best_model = random_search.best_estimator_
print("Best parameters found:", random_search.best_params_)

# Test set
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
preds = best_model.predict(test)
preds = np.expm1(preds)
preds = np.clip(preds, 0, None)

# Submission
submission = pd.DataFrame({
    'datetime': test_datetime,
    'count': preds
})
submission.to_csv('submission_lgbm_randomsearch.csv', index=False)
print("Submission saved as 'submission_lgbm_randomsearch.csv'")

# Plotting
plt.figure(figsize=(14, 5))
sns.lineplot(data=submission, x='datetime', y='count')
plt.title('LightGBM Predictions Over Time (RandomizedSearchCV)')
plt.xlabel('Datetime')
plt.ylabel('Predicted Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
