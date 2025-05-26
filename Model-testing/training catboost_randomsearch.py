import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_log_error


train = pd.read_csv('../datasets/train.csv')

# Splitting datetime to hour,day,weekday,year and month
train['datetime'] = pd.to_datetime(train['datetime'])
train['hour'] = train['datetime'].dt.hour
train['day'] = train['datetime'].dt.day
train['weekday'] = train['datetime'].dt.weekday
train['year'] = train['datetime'].dt.year
# train['month'] = train['datetime'].dt.month
# Increases error as month is a categorical variable here. OHE will give similar results


train = train.drop(['datetime', 'casual', 'registered', 'month'], axis=1, errors='ignore')


X = train.drop('count', axis=1)
y = np.log1p(train['count']) #Log transforming count for lower rmsle score


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, np.maximum(0, y_pred)))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

#CatBoostRegressor
#documentation:https://catboost.ai/docs/en/references/training-parameters/
cat_model = CatBoostRegressor(
    silent=True,#no logging output given
    random_state=42
)

# Parameter grid for random search
param_dist = {
    'iterations': [500, 1000, 1500],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8, 10],#how deep a tree should go
    'l2_leaf_reg': [1, 3, 5, 7, 9],#to prevent overfitting
    'bagging_temperature': [0.5, 1.0, 1.5],#randomness for bagging
    'border_count': [32, 64, 128]
}

# Randomized Search
random_search = RandomizedSearchCV(
    estimator=cat_model,#what model to tune
    param_distributions=param_dist,#use the parameters combo given above
    n_iter=30,#Number of random iterations
    scoring=rmsle_scorer,
    cv=3,#number of folds for cross verification
    verbose=2,
    n_jobs=-1,#to use all the cores
    random_state=42
)

random_search.fit(X, y) #fit the model
best_model = random_search.best_estimator_#gives us the best estimators given by randomsearch cv
print("Best parameters found:", random_search.best_params_)


test = pd.read_csv('../datasets/test.csv')
test_datetime = test['datetime']
test['datetime'] = pd.to_datetime(test['datetime'])

test['hour'] = test['datetime'].dt.hour
test['day'] = test['datetime'].dt.day
test['weekday'] = test['datetime'].dt.weekday
test['year'] = test['datetime'].dt.year
# test['month'] = test['datetime'].dt.month  # Not used

test = test.drop(['datetime', 'month'], axis=1, errors='ignore')


test_preds = best_model.predict(test)#predict the value
test_preds = np.expm1(test_preds) #inverse log the value back
test_preds = np.clip(test_preds, 0, None) #no negative numbers will be shown

# Submission
submission = pd.DataFrame({
    'datetime': test_datetime,
    'count': test_preds
})

# Plot predictions
plt.figure(figsize=(14, 5))
sns.lineplot(data=submission, x='datetime', y='count')
plt.title('CatBoost Predictions Over Time')
plt.xlabel('Datetime')
plt.ylabel('Predicted Count')
plt.tight_layout()
plt.show()

#scatter
plt.figure(figsize=(14, 5))
sns.scatterplot(data=submission, x='datetime', y='count', alpha=0.5, s=10)
plt.title('Predicted Rentals Scatter Over Time')
plt.xlabel('Datetime')
plt.ylabel('Predicted Count')
plt.tight_layout()
plt.show()

#rollling average
submission['rolling_mean'] = submission['count'].rolling(window=24).mean()  # Rolling over 1 day if hourly

plt.figure(figsize=(14, 5))
sns.lineplot(data=submission, x='datetime', y='rolling_mean', label='24-hour Rolling Mean')
sns.lineplot(data=submission, x='datetime', y='count', alpha=0.3, label='Original')
plt.title('Rolling Average of Predicted Rentals')
plt.xlabel('Datetime')
plt.ylabel('Predicted Count')
plt.legend()
plt.tight_layout()
plt.show()

#histograph
plt.figure(figsize=(10, 5))
sns.histplot(submission['count'], bins=30, kde=True, color='teal')
plt.title('Distribution of Predicted Counts')
plt.xlabel('Predicted Count')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

submission.to_csv('submission_catboost_randomsearch.csv', index=False)
print("Submission saved as 'submission_catboost_randomsearch.csv'")
