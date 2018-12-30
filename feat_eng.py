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
from sklearn.preprocessing import StandardScaler,OneHotEncoder


path = "data"
train = pd.read_csv(path + "/train_V2.csv")
train_cats(train)
train = train[np.isfinite(train['winPlacePerc'])]

# players in match
train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')

# normalize kills and damageDealt
train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)
train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)

# create variable healsAndBoosts(combined heals and boosts), and totalDistance-
# walking, riding, swimming distances
train['healsAndBoosts'] = train['heals']+train['boosts']
train['totalDistance'] = train['walkDistance']+train['rideDistance']+train['swimDistance']

# boosts, heals, kills, per walk distance
train['boostsPerWalkDistance'] = train['boosts']/(train['walkDistance']+1)
train['boostsPerWalkDistance'].fillna(0, inplace=True)
train['healsPerWalkDistance'] = train['heals']/(train['walkDistance']+1)
train['healsPerWalkDistance'].fillna(0, inplace=True)
train['healsAndBoostsPerWalkDistance'] = train['healsAndBoosts']/(train['walkDistance']+1)
train['healsAndBoostsPerWalkDistance'].fillna(0, inplace=True)
train['killsPerWalkDistance'] = train['kills']/(train['walkDistance']+1)
train['killsPerWalkDistance'].fillna(0, inplace=True)
train[['kills', 'walkDistance', 'rideDistance', 'killsPerWalkDistance',
    'winPlacePerc']].sort_values(by='killsPerWalkDistance').tail(10)
train['killsPerTotalDistance'] = train['kills']/(train['totalDistance']+1)

# variabl for teams
train['team'] = [1 if i>50 else 2 if (i>25 & i<=50) else 4 for i in train['numGroups']]
train.shape

train.to_csv("output/train_clean2.csv")
