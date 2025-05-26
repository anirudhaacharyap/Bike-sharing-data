import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_log_error
from scipy.stats import uniform, randint

train = pd.read_csv('train.csv')


train['datetime'] = pd.to_datetime(train['datetime'])
train['hour'] = train['datetime'].dt.hour
train['day'] = train['datetime'].dt.day
train['weekday'] = train['datetime'].dt.weekday
train['month'] = train['datetime'].dt.month
train['year'] = train['datetime'].dt.year


train = train.drop(['datetime', 'casual', 'registered', 'month'], axis=1)


X = train.drop('count', axis=1)
y = np.log1p(train['count'])


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, np.maximum(0, y_pred)))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

# Randomized search parameter space
param_distributions = {
    'n_estimators': randint(100, 400),
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.6, 0.4),  # range: 0.6 to 1.0
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 2),
    'min_child_weight': randint(1, 10)
}

# Initialize XGBoost model
xgb = XGBRegressor(random_state=42, objective='reg:squarederror')


random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_distributions,
    scoring=rmsle_scorer,
    cv=3,
    n_iter=50,
    verbose=2,
    random_state=42,
    n_jobs=-1
)


random_search.fit(X, y)
best_model = random_search.best_estimator_
print("Best Parameters Found:", random_search.best_params_)

# Load and preprocess test data
test = pd.read_csv('test.csv')
test_datetime = test['datetime']
test['datetime'] = pd.to_datetime(test['datetime'])
test['hour'] = test['datetime'].dt.hour
test['day'] = test['datetime'].dt.day
test['weekday'] = test['datetime'].dt.weekday
test['year'] = test['datetime'].dt.year
test = test.drop(['datetime'], axis=1)

test_preds = best_model.predict(test)
test_preds = np.expm1(test_preds)
test_preds = np.clip(test_preds, 0, None)


submission = pd.DataFrame({
    'datetime': test_datetime,
    'count': test_preds
})

submission.to_csv('submission_xgboost_randomsearch.csv', index=False)
print("Submission saved as 'submission_xgboost_randomsearch.csv'")

# Plotting
plt.figure(figsize=(14, 5))
sns.lineplot(data=submission, x='datetime', y='count')
plt.title('XGBoost Predictions Over Time')
plt.xlabel('Datetime')
plt.ylabel('Predicted Count')
plt.tight_layout()
plt.show()
