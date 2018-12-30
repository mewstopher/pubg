import sys
sys.path.append('/anaconda3/envs/fastai-cpu/lib/python3.6/site-packages')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from fastai.imports import *
from fastai.structured import *
from sklearn.metrics import mean_squared_error
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from IPython.display import display
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor


path = "../output"
data = pd.read_csv(path + '/train_clean2.csv')
train_cats(data)
data = data[np.isfinite(data['winPlacePerc'])]
df, y, nas = proc_df(data, 'winPlacePerc')
X_train, X_test, y_train, y_test = train_test_split(df, y)

kfold = 5

set_rf_samples(500000)
rf_pipe = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestRegressor(random_state=7))])
rf_params = {'rf__n_estimators': [40,100,150],
        'rf__min_samples_leaf':[1, 3, 5]}
rf_grid = GridSearchCV(rf_pipe, param_grid=rf_params, cv=kfold,n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)
print('training complete')

print('rf scores')
print(rf_grid.best_params_)
print(rf_grid.best_score_)
