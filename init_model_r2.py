import sys
sys.path.append('/anaconda3/envs/fastai-cpu/lib/python3.6/site-packages')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier
from fastai.imports import *
from fastai.structured import *
from sklearn.metrics import mean_squared_error
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# initial random forest using semi cleaned data
# train score: .005
# test score: .007

def print_score(m):
    res = [mean_squared_error(m.predict(X_train), y_train), mean_squared_error(m.predict(X_test), y_test)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)

path = "output"
train = pd.read_csv(path + "/train_clean.csv")
train_cats(train)
train = train[np.isfinite(train['winPlacePerc'])]
df, y, nas = proc_df(train, 'winPlacePerc')
X_train, X_test, y_train, y_test = train_test_split(df, y)
print('loading complete')

set_rf_samples(500000)
m = RandomForestRegressor(random_state=8, n_jobs=-1, verbose=10)
m.fit(X_train, y_train)

print('training complete')
print_score(m)

fi = rf_feat_importance(m, X_train)
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi[:20]); 
