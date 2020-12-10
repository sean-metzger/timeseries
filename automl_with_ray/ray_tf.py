import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error


pjme = pd.read_csv('/Users/sean/Downloads/energy_forecast/PJME_hourly.csv',
                  index_col=[0], parse_dates=[0])

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


pjme_x = extract_feats(pjme_x)
pjme_x.head()

split_date = '01-01-2015'

x_tr = pjme_x.loc[pjme_x.index < split_date]
y_tr = pjme_y.loc[pjme_y.index < split_date]

x_te = pjme_x.loc[pjme_x.index >= split_date]
y_te = pjme_y.loc[pjme_y.index >= split_date]


from xgboost import XGBRegressor

model = XGBRegressor()

# Make the dataset
x_te = x_te[10:]
y_te = y_te[10:]

x_te.values.shape

import ray
from ray import tune
# ray.services.get_node_ip_address = lambda: '127.0.0.1'
ray.init(local_mode=True,num_cpus=1, num_gpus=0)

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

class MyModel(Model):
    def __init__(self, hid1, hid2): 
        super(MyModel, self).__init__()
        self.d1 = Dense(hid1, activation='relu')
        self.d2 = Dense(hid2, activation='relu')
        self.d3 = Dense(1)
        
    def call(self, x): 
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)

def load_data(): 
    return (x_tr.values, y_tr.values), (x_te.values, y_te.values)

class TSTrainer(tune.Trainable): 
    def setup(self, config): 
        
        import tensorflow as tf
        
        (x_train, y_train), (x_test, y_test) = load_data()
        
        self.train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
        self.test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
        
        self.model = MyModel(hid1=config['hid1'], 
                            hid2=config['hid2'])
        
        self.loss_func = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(lr=config['lr'])
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss= tf.keras.metrics.Mean(name='test_loss')
        self.train_mae = tf.keras.metrics.MeanAbsoluteError()
        self.test_mae = tf.keras.metrics.MeanAbsoluteError()
        
        @tf.function
        def train_step(x, y): 
            with tf.GradientTape() as tape:
                pred = self.model(x)
                loss = self.loss_func(y, pred)
                self.train_mae(pred, y)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))
            self.train_loss(loss)
            

        @tf.function
        def test_step(x, y): 
            pred = self.model(x)
            loss = self.loss_func(y, pred)
            self.test_loss(loss)
            self.test_mae(pred, y)
        
        self.tf_train_step = train_step
        self.tf_test_step = test_step
        
    def step(self): 
        self.train_loss.reset_states()
        self.test_loss.reset_states()
        self.train_mae.reset_states()
        self.test_mae.reset_states()
        
        for i, (x, y) in enumerate(self.train_ds): 
            self.tf_train_step(x, y)
        for i, (x, y) in enumerate(self.test_ds): 
            self.tf_test_step(x, y)
            
        return{ 
            'epoch': self.iteration, 
            'loss': self.train_loss.result().numpy(), 
            'test_loss': self.test_loss.result().numpy(),
            'mae': self.train_mae.result().numpy(),
            'test_mae': self.test_mae.result().numpy()
        }
        
        
    

load_data()

from ray.tune.schedulers import AsyncHyperBandScheduler
sched = AsyncHyperBandScheduler(max_t=30, grace_period=20)




analysis = tune.run(
    TSTrainer, 
    metric='test_loss',
    scheduler=sched,
    mode='min',
    stop={'training_iteration':5},
    verbose=2,
    num_samples=3,
    config={'hid1':tune.choice([32, 64, 128]), 
           'hid2':tune.choice([32, 64, 128]),
            'lr':tune.choice([1e-3, 1e-5])
           }
)
analysis.best_config