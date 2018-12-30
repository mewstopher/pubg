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


path = "output"
data = pd.read_csv(path + '/train_clean2.csv')
train_cats(data)

data = data[np.isfinite(data['winPlacePerc'])]
df, y, nas = proc_df(data, 'winPlacePerc')
X_train, X_test, y_train, y_test = train_test_split(df, y)

random_state = 2
classifiers = []

classifiers.append(RandomForestRegressor(random_state=random_state))
classifiers.append(GradientBoostingRegressor(random_state=random_state))
classifiers.append(XGBRegressor(random_state=random_state))
classifiers.append(DecisionTreeRegressor(random_state=random_state))
classifiers.append(AdaBoostRegressor(DecisionTreeRegressor(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(ExtraTreesRegressor(random_state=random_state))
classifiers.append(KNeighborsRegressor())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = y_train,
        scoring='accuracy', cv=5, n_jobs=4))
sklearn.metrics.SCORERS.keys()
cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,
    "Algorithm":['RandomForest', 'GradientBoostingRegressor',
        'XGB', 'DecisionTreeRegressor', 'adaboost', 'ExtraTrees',
        'Knn']})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")

cv_res
