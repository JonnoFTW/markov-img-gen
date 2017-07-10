from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import Callback
import numpy as np
import pyprind
from .utils import readwav, train_test_split
import sounddevice as sd

output_seconds = 5
n_gram = 2
max_len = 1
fname = 'Night_And_Day.flac'
fs, data, outname = readwav(fname)
data = data[fs * 30:fs * 60]
batch_size = 1
nb_epoch = 1
lookback = 4
model = Sequential()
model.add(LSTM(128, batch_input_shape=(batch_size, max_len, lookback), return_sequences=True, stateful=True, activation='relu'))
model.add(LSTM(96,  return_sequences=False, stateful=True, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(loss='mse', optimizer='adam')

steps = 1
print("steps:", steps)
features = data[:-steps]
labels = data[steps:]


# Each individual input record should be a `lookback` number of records, the label is the next excpected record
def chunks(x, y, n, lb):
    """Yield successive n-sized chunks from l."""
    for i in range(lb, len(x), n):
        yield x[i - lb:i], y[i]


tts = train_test_split(features, labels, test_size=0.5)

X_train = tts[0].reshape(tts[0].size, 1)
X_test = tts[1].reshape(tts[1].size, 1)
Y_train = tts[2].astype(np.float64).reshape(tts[2].size, 1)
Y_test = tts[3].astype(np.float64).reshape(tts[3].size, 1)

# x_tr_chop = X_train.size % max_len
# x_te_chop = X_test.size % max_len
# y_tr_chop = Y_train.size % max_len
# y_te_chop = Y_test.size % max_len
#
# X_train = X_train[:-x_tr_chop]  # .reshape(X_train.size // max_len, max_len, 1)
# Y_train = Y_train[:-y_tr_chop]  # .reshape(Y_train.size - y_tr_chop, 1)
# X_test = X_test[:-x_te_chop]  # .reshape(X_test.size // max_len, max_len, 1)
# Y_test = Y_test[:-y_te_chop]  # .reshape(Y_test.size - y_te_chop, 1)

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

print('Train...')

for i in range(nb_epoch):
    print("Epoch:", i)
    # mean_tr_acc = []
    mean_tr_loss = []
    prog = pyprind.ProgBar(X_train.size - lookback )
    c = 0
    for x_chunk, y_chunk in chunks(X_train, Y_train, batch_size, lookback):
        tr_loss = model.train_on_batch(x_chunk.reshape(1, 1, lookback), y_chunk.reshape(1, 1))
        prog.update()
        c+=1
        # mean_tr_acc.append(tr_acc)
        mean_tr_loss.append(tr_loss)
    # print('accuracy training = {}'.format(np.mean(mean_tr_acc)))
    print('loss training = {}'.format(np.mean(mean_tr_loss)))
    print("Batches run:", c)
    print('___________________________________')

print()
# generate some sound using the model and random initilisation from raw data
state_idx = np.random.randint(0, data.size - lookback)
state = data[state_idx:state_idx + lookback]
output_frames = fs * output_seconds
generated = np.zeros(output_frames, dtype=np.int16)
prog = pyprind.ProgBar(output_frames)
for i in range(output_frames):
    generated[i] = model.predict(state.reshape(1, 1, lookback))[0][0]
    state = np.roll(state, -1)
    state[-1] = generated[i]
    prog.update()
print("\nPlaying")
all_frames = np.concatenate((data, generated))
sd.play(all_frames, fs, blocking=True)
print("Finished playing")
