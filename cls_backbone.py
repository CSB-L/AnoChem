# import tempfile
import os

import tensorflow as tf
import numpy as np

from tensorflow import keras


def dnn_cls(lin_dims=[2048,512,256,128,32,1],dropout=0.2,act='relu',last_act='sigmoid'):
    _seq_ = []
    for _idx, _dim in enumerate(lin_dims[1:]):
        if _idx == 0:
            _seq_.append(tf.keras.layers.Dense(_dim,activation=act,input_dim=lin_dims[0]))
        elif _idx == len(lin_dims)-2:
            _seq_.append(tf.keras.layers.Dense(_dim,activation=last_act))
            break
        else:
            _seq_.append(tf.keras.layers.Dense(_dim,activation=act))
        if type(dropout)==float:
            _seq_.append(tf.keras.layers.Dropout(dropout))
    model = tf.keras.models.Sequential(_seq_)
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Dense(512, activation=act,input_dim=2048),
#         tf.keras.layers.Dropout(dropout),
#         tf.keras.layers.Dense(256, activation=act),
#         tf.keras.layers.Dropout(dropout),
#         tf.keras.layers.Dense(128, activation=act),
#         tf.keras.layers.Dropout(dropout),
#         tf.keras.layers.Dense(32, activation=act),
#         tf.keras.layers.Dropout(dropout),
#         tf.keras.layers.Dense(1)
#     ])
    
    return model


def compile_model(model,**compile_option):
    # Defaut settings
    _param = compile_option.copy()
    if 'optimizer' not in compile_option:
        if 'learning_rate' in compile_option:
            opt = tf.keras.optimizers.Adam(learning_rate=compile_option['learning_rate'])
            _param.pop('learning_rate')
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    else:
        opt = compile_option['optimizer']
        _param.pop('optimizer')
        
    if 'loss' not in compile_option:
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        loss = compile_option['loss']
        _param.pop('loss')
    if 'metrics' not in compile_option:
        metrics = ['accuracy',loss]
    else:
        metrics = compile_option['metrics']
        _param.pop('metrics')
        
    model.compile(optimizer=opt,loss=loss,metrics=metrics)

    
# def training():
def train(model,
          trainx,trainy,
          validx,validy,
          batch_size=16,ckpt_dir=False,max_epoch=1000,patience=False):
    callbacks = []
    if ckpt_dir:
        out_d = os.path.abspath(ckpt_dir)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                out_d+'/tmp_model.{epoch:04d}.hdf5',
                monitor='val_loss',
                verbose=0, save_best_only=False,
                save_weights_only=False, mode='auto', save_freq=1))
    if patience:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience))
#     train_history=model.fit(
    return model.fit(
        trainx,trainy,
        batch_size=batch_size,
        epochs=max_epoch,
        validation_data=(validx,validy),
        callbacks=callbacks,
    )
#     return train_history
    
    
def evaluate(model,x,y):
    return model.evaluate(x,y)
    
def predict(model,x):
    return model.predict(x)

def save_model(model,f):
    if not f.endswith('.h5'):
        _f = f+'.h5'
    else:
        _f = f
    model.save(_f)


def load_model(f,**compile_option):
    model = tf.keras.models.load_model(f,compile=False)
    compile_model(model,**compile_option)
    return model



############### Appendix ###############
from sklearn.metrics import accuracy_score
import re

class WrapModel:
    def __init__(self,**params):
        self.nodes=[2048,512,256,128,32,1]
        self.dropout=0.5
        self.params = dict(
            nodes=[2048,512,256,128,32,1],
            dropout=0.5,
            learning_rate=1e-4,
            batch_size=128,
            validation_split=0.2,
            max_epoch=100,
            patience=10,
        )
        self.params.update(params)
        _f_n='.'.join([str(i) for i in self.params['nodes']])
        _f_n+='.'+re.sub('\W','_','.'.join([str(i) for i in self.params.values() if type(i) not in [set,list,tuple,dict]]))
        self.ckpt_dir = f'./tmp_dir/{_f_n}'
        self.model=None
        self.train_history={}
        self.callbacks=[]
        
    def get_params(self):
        return self.params
    
    def set_params(self,**params):
        self.params.update(params)
        _f_n='.'.join([str(i) for i in self.params['nodes']])
        _f_n+='.'+re.sub('\W','_','.'.join([str(i) for i in self.params.values() if type(i) not in [set,list,tuple,dict]]))
        self.ckpt_dir = f'./tmp_dir/{_f_n}'
        self.model=None
        
    def score(self,X,y,sample_weight=None):
        y_pred = (self.model.predict(X)>0.5).astype('int32')
        return accuracy_score(y_true=y,y_pred=y_pred)
    
    def fit(self,X,y):
        self.model = cb.dnn_cls(lin_dims=self.params['nodes'],dropout=self.params['dropout'])
        cb.compile_model(self.model,learning_rate=self.params['learning_rate'])
        
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=self.params['patience']),
            tf.keras.callbacks.ModelCheckpoint(
                self.ckpt_dir+'/tmp_model.{epoch:04d}.hdf5',
                monitor='val_loss',
                verbose=0, save_best_only=False,
                save_weights_only=False, mode='auto', save_freq=1)
        ]
        self.train_history=self.model.fit(
            X,y,
            batch_size=self.params['batch_size'],
            epochs=self.params['max_epoch'],
            validation_split=self.params['validation_split'],
#             validation_data=(validx,validy),
            callbacks=self.callbacks,
        )
        with open(self.ckpt_dirt_dir+'/train_history,pkl','wb') as f:
            pickle.dump(f,self.train_history)
        
    def predict(self,X):
        return (self.model.predict(X)>0.5).astype('int32')
    
    def predict_proba(self,X):
        pos = self.model.predict(X).reshape(-1,1)
        neg = 1.-pos
        proba = np.concatenate((neg,pos),axis=1)
        return proba

