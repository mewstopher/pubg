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
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(df)

# exploratory data analysis

path = "data"
train = pd.read_csv(path + "/train_V2.csv")
train_cats(train)
train = train[np.isfinite(train['winPlacePerc'])]
df, y, nas = proc_df(train, 'winPlacePerc')

display_all(train.tail().T)



train.describe().T



g = sns.heatmap(train[:].corr(),annot=True,  cmap = "coolwarm")

# winpoints: bimodel distribution. one at zero, one at 1500
g = sns.distplot(train['winPoints'], color="m")

# weaponsAcquired: log transform
g = sns.distplot(train['weaponsAcquired'], color="m")

train["weaponsAcquired"] = train["weaponsAcquired"].map(lambda i: np.log(i) if i > 0 else 0)

# walk distance
g = sns.distplot(train['walkDistance'], color="m")
train['zero_walked'] = train['walkDistance'].map(lambda i: 1 if i==0 else 0)

g = sns.factorplot(x="zero_walked",y="winPlacePerc",data=train,kind="bar",
    palette = "muted")

# vehicle destroys

g = sns.factorplot(x= 'vehicleDestroys', y ='winPlacePerc',data = train,
    kind='bar',palette = 'muted')

# team kills

g = sns.factorplot(x= 'teamKills', y ='winPlacePerc',data = train,
    kind='bar',palette = 'muted')

train['teamkill_lo'] = train['teamKills'].map(lambda s: 1 if s < 6 else 0)
train['teamkill_hi'] = train['teamKills'].map(lambda s:1 if s > 5 else 0)

# swim distance
# log transform
g = sns.distplot(train['swimDistance'], color = 'm')

train["swimDistance"] = train["swimDistance"].map(lambda i: np.log(i) if i > 0 else 0)

# road kills
g = sns.factorplot(x= 'roadKills', y ='winPlacePerc',data = train,
    kind='bar',palette = 'muted')

# ride distance
train["rideDistance"] = train["rideDistance"].map(lambda i: np.log(i) if i > 0 else 0)
g = sns.distplot(train['rideDistance'], color = 'm')

# revives
g = sns.distplot(train['revives'], color = 'm')

train["revives"] = train["revives"].map(lambda i: np.log(i) if i > 0 else 0)

# rankpoints
g = sns.distplot(train['rankPoints'], color ='m')

# numGroups

g = sns.distplot(train['numGroups'], color ='m')
train['group_xs'] = train['numGroups'].map(lambda i: 1 if i < 20 else 0)
train['group_s'] = train['numGroups'].map(lambda i: 1 if 20< i < 40 else 0)
train['group_m'] = train['numGroups'].map(lambda i: 1 if 40<= i <= 60 else 0)
train['group_lg'] = train['numGroups'].map(lambda i: 1 if 60 < i <=80 else 0)
train['group_xl'] = train['numGroups'].map(lambda i: 1 if i > 80 else 0)

g = sns.factorplot(x='group_xs', y = 'winPlacePerc', data = train,
    kind='bar', palette='muted')

g = sns.factorplot(x='group_s', y = 'winPlacePerc', data = train,
    kind='bar', palette='muted')

g = sns.factorplot(x='group_m', y = 'winPlacePerc', data = train,
    kind='bar', palette='muted')

g = sns.factorplot(x='group_lg', y = 'winPlacePerc', data = train,
    kind='bar', palette='muted')

g = sns.factorplot(x='group_xl', y = 'winPlacePerc', data = train,
    kind='bar', palette='muted')


# max place
# displays same groups as numGroups ( 20-40, 40-60, 80-100)
g = sns.distplot(train['maxPlace'], color ='m')

# matchtype
# get dummy variables for match types
train = pd.get_dummies(train, columns = ["matchType"])

# longest kill
g = sns.distplot(train['longestKill'], color='m')

train["longestKill"] = train["longestKill"].map(lambda i: np.log(i) if i > 0 else 0)

# killstreaks
g = sns.factorplot(x='killStreaks', y = 'winPlacePerc', data = train, kind='bar',
    palette='muted')

train['kstreak_lo'] = train['killStreaks'].map(lambda i: 1 if i == 0 else 0)
train['kstreak_med'] = train['killStreaks'].map(lambda i:1 if 0< i < 18 else 0)
train['kstreak_hi'] = train['killStreaks'].map(lambda i:1 if i > 16 else 0)

# kills

g = sns.factorplot(x='kills', y='winPlacePerc', data=train)
g = sns.distplot(train['kills'], color='m')

train["kills"] = train["kills"].map(lambda i: np.log(i) if i > 0 else 0)

# headshotkills
g = sns.distplot(train['headshotKills'], color='m')

# boosts
g = sns.distplot(train['boosts'], color='m')
# assists
g = sns.distplot(train['assists'], color='m')

os.listdir()
train.to_csv("output/train_clean.csv")
