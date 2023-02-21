# import tempfile
import os

import tensorflow as tf
import numpy as np

from tensorflow import keras



def dnn_cls(lin_dims=[2048,512,256,128,32,1],dropout=0.2,act='relu'):
    _seq_ = []
    for _idx, _dim in enumerate(lin_dims[1:]):
        if _idx == 0:
            _seq_.append(tf.keras.layers.Dense(_dim,activation=act,input_dim=lin_dims[0]))
        elif _idx == len(lin_dims)-2:
            _seq_.append(tf.keras.layers.Dense(_dim))
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
def train(model,trainx,trainy,validx,validy,ckpt_dir,batch_size=4,max_epoch=1000,patience=10):
    out_d = os.path.abspath(ckpt_dir)
    train_history=model.fit(
        trainx,trainy,
        batch_size=batch_size,
        epochs=max_epoch,
        validation_data=(validx,validy),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy',patience=patience),
        tf.keras.callbacks.ModelCheckpoint(out_d+'/tmp_model.{epoch:04d}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq=1)

        ],
    )
    return train_history
    
    
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

