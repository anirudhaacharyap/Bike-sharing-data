import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_log_error,make_scorer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('../datasets/train.csv')


train['datetime'] = pd.to_datetime(train['datetime'])
train['hour'] = train['datetime'].dt.hour
train['day'] = train['datetime'].dt.day
train['weekday'] = train['datetime'].dt.weekday
train['month'] = train['datetime'].dt.month
train['year'] = train['datetime'].dt.year


train = train.drop(['datetime', 'casual', 'registered','month'], axis=1)


X = train.drop('count', axis=1)
y = np.log1p(train['count'])

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, np.maximum(0,y_pred)))

param_grid = {'n_estimators': [ 50, 100],
              'max_depth': [10,20],
              'min_samples_split': [2, 5]}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    scoring=make_scorer(rmsle, greater_is_better=False),
    cv=3,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X, y)
model_best = grid_search.best_estimator_
print("The Best parameters are:",grid_search.best_params_)

test = pd.read_csv('../datasets/test.csv')

test_datetime = test['datetime']
test['datetime'] = pd.to_datetime(test['datetime'])
test['hour'] = test['datetime'].dt.hour
test['day'] = test['datetime'].dt.day
test['weekday'] = test['datetime'].dt.weekday
test['month'] = test['datetime'].dt.month
test['year'] = test['datetime'].dt.year


test = test.drop(['datetime','month'], axis=1)


test_preds=model_best.predict(test)
test_preds=np.expm1(test_preds)
test_preds=np.clip(test_preds,0,None)


submission = pd.DataFrame({
    'datetime': test_datetime,
    'count': test_preds
})

plt.figure(figsize=(14, 5))
sns.lineplot(data=submission, x='datetime', y='count')
plt.title('Predicted Bike Rentals Over Time')
plt.xlabel('Datetime')
plt.ylabel('Predicted Count')
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 5))
sns.scatterplot(data=submission, x='datetime', y='count', s=10, color='green')
plt.title('Scatter Plot: Predicted Bike Rentals Over Time')
plt.xlabel('Datetime')
plt.ylabel('Predicted Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


submission.to_csv('submission_random_forest_nomonth_noWindspeed_log.csv', index=False)
print("Submission saved as 'submission_random_forest_GridSearch.csv'")
