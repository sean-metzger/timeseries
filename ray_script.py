import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
import ray
from ray import tune


pjme = pd.read_csv('gs://seantimeseries/PJME_hourly.csv',
                  index_col=[0], parse_dates=[0])

plt.style.use('fivethirtyeight')

def extract_window(dataframe): 
    cols = []
    for i in range(10, -1, -1): 
        cols.append(dataframe.shift(i))
        
    df= pd.concat(cols, axis=1)
    df.columns = ['t-%d' %k for k in range(10, -1, -1)]
    df = df.dropna()
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    return x, y

pjme_x, pjme_y = extract_window(pjme)

def extract_feats(df): 
    df['date'] = df.index.day
    df['month']  = df.index.month
    df['hour'] = df.index.hour
    df['day_of_year'] = df.index.dayofyear
    df['day_of_week'] = df.index.dayofweek
    df['week_of_year'] = df.index.weekofyear
    return df

split_date = '01-01-2015'

pjme_x = extract_feats(pjme_x)

x_tr = pjme_x.loc[pjme_x.index < split_date]
y_tr = pjme_y.loc[pjme_y.index < split_date]

x_te = pjme_x.loc[pjme_x.index >= split_date]
y_te = pjme_y.loc[pjme_y.index >= split_date]

# More advanced Ray tuning
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler

def train_xgb_asha(config): 
    train_set = xgb.DMatrix(x_tr, label=y_tr)
    test_set = xgb.DMatrix(x_te, label=y_te)
    
    xgb.train(
        config,
        train_set, 
        100,
        evals=[(test_set, 'eval')], 
        verbose_eval=False, 
        early_stopping_rounds=10,
        callbacks=[TuneReportCheckpointCallback(filename='model.xgb')])
    
config={ 
    'objective':'reg:squarederror', 
    "eval_metric":['mae'],
    'max_depth':tune.randint(1,9),
    'min_child_weight':tune.choice([1, 2, 3]), 
    'subsample':tune.uniform(0.5, 1.0), 
    'eta':tune.uniform(1e-4, 1e-1),
}
    
scheduler=ASHAScheduler(
    time_attr='training_iteration',
    max_t=1000, 
    grace_period=1, 
    reduction_factor=2
)
    
analysis = tune.run(
    train_xgb_asha, 
    metric='eval-mae', 
    mode='min', 
    resources_per_trial={'cpu':1}, 
    config=config, 
    num_samples=10, 
    scheduler=scheduler
)