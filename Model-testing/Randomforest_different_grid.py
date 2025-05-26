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


'''file = train.drop(columns=['datetime'])
correlation_matrix = file.corr()
sns.heatmap(correlation_matrix, cmap='hot', annot=True)
plt.show()'''




train = train.drop(['datetime', 'casual', 'registered'], axis=1)


X = train.drop('count', axis=1)
y = np.log1p(train['count'])


def rmsle(y_true, y_pred, convertExp=True):
    if convertExp:
        y_true = np.exp(y_true)
        y_pred = np.exp(y_pred)

    log_true = np.nan_to_num(np.log(y_true + 1))
    log_pred = np.nan_to_num(np.log(y_pred + 1))

    output = np.sqrt(np.mean((log_true - log_pred) ** 2))
    return output
rmsle_score = make_scorer(rmsle, greater_is_better=False)

randomforest = RandomForestRegressor()

rf_params = {'random_state': [42], 'n_estimators': [10, 20, 140]}
gridsearch_random_forest = GridSearchCV(estimator=randomforest,
                                        param_grid=rf_params,
                                        scoring=rmsle_scorer,
                                        cv=5)

log_y = np.log(y)
gridsearch_random_forest.fit(X_train, log_y)
print(f'Best Parameter: {gridsearch_random_forest.best_params_}')

test = pd.read_csv('../datasets/test.csv')

test_datetime = test['datetime']
test['datetime'] = pd.to_datetime(test['datetime'])
test['hour'] = test['datetime'].dt.hour
test['day'] = test['datetime'].dt.day
test['weekday'] = test['datetime'].dt.weekday
test['month'] = test['datetime'].dt.month
test['year'] = test['datetime'].dt.year


test = test.drop(['datetime'], axis=1)


test_preds=model_best.predict(test)
test_preds=np.expm1(test_preds)
test_preds=np.clip(test_preds,0,None)


submission = pd.DataFrame({
    'datetime': test_datetime,
    'count': test_preds
})

submission.to_csv('submission_random_forest_OHE_Log.csv', index=False)
print("Submission saved as 'submission_random_forest_GridSearch.csv'")
