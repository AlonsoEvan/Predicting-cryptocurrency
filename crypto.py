import pandas as pd
from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint
import time
from sklearn import preprocessing

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "BTC-USD"
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


def preprocess_df(df):
    df = df.drop(f'{RATIO_TO_PREDICT}_future', axis = 1)
    
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace = True)
            df[col] = preprocessing.scale(df[col].values)
            
    df.dropna(inplace = True)
    
    
    
    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
    
    random.shuffle(sequential_data)
    
    buys = []
    sells = []
    
    for seq, target in sequential_data:
        if target == 0: #if it's "not buy"
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
            
    random.shuffle(buys)
    random.shuffle(sells)
    
    lower = (min(len(buys), len(sells)))
    
    buys = buys[:lower]
    sells = sells[:lower]
    
    sequential_data = buys + sells
    random.shuffle(sequential_data)
    
    
    X = []
    y = []
    
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
    
    return np.array(X), y


main_df = pd.DataFrame()

ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]

for ratio in ratios:
    dataset = f'datasets/{ratio}.csv'
    df = pd.read_csv(dataset, names = ['time', 'low', 'high', 'open', 'close', 'volume'])  # read in specific file
    
    df.rename(columns = {"close" : f"{ratio}_close", "volume" : f"{ratio}_volume"}, inplace  = True)
    
    df.set_index("time", inplace = True)
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]
    
    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)
        
main_df.fillna(method = 'ffill', inplace = True)
main_df.dropna(inplace = True)


main_df[f'{RATIO_TO_PREDICT}_future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df[f'{RATIO_TO_PREDICT}_future']))

times = sorted(main_df.index.values)
last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

train_x = np.array(train_x)
train_y = np.array(train_y)
validation_x = np.array(validation_x)
validation_y = np.array(validation_y)




model = Sequential()

model.add(LSTM(128, input_shape = (train_x.shape[1:]), activation = 'relu',  return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape = (train_x.shape[1:]), activation = 'relu',  return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape = (train_x.shape[1:]), activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation = 'softmax'))


opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)


model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)


history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    verbose = 2,
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir="logs\{}".format(NAME), histogram_freq=1, profile_batch = 100000000)])