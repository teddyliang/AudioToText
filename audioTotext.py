"""
This project aims to build a simple audio to text artificial intelligence using keras and a convulutional neural network.
The data is taken from the Kaggle "TensorFlow Speech Recognition Challenge."
It is trained on the words "yes, no, up, down, left, right, on off, stop, go."
"""
#importing libraries
import os
import keras
import librosa   #for audio processing
import librosa.display
import seaborn as sns

import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile #for audio processing
from scipy import signal
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D, Activation, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Adagrad, RMSprop, Adam, Nadam
from keras import backend as K
from sklearn.model_selection import train_test_split
from matplotlib import pyplot 
from keras.models import load_model
import random
import sounddevice as sd
import soundfile as sf
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import pandas as pd


#Data exploration and visualization

#displaying an audio signal
audioTrainPath = '/Users/teddy/Downloads/tensorflow-speech-recognition-challenge/train/audio/'
sampleRate, samples = wavfile.read(str(audioTrainPath) + 'yes/0a7c2a8d_nohash_0.wav')


#function for getting frequencies, times, and a spectrogram for plotting
def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

freqs, times, spectrogram = log_specgram(samples, sampleRate)

#plotting a sample wave
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Audio wave of ' + '0a7c2a8d_nohash_0.wav')
ax1.set_xlabel('time')
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, sampleRate/len(samples), sampleRate), samples)

ax2 = fig.add_subplot(212)
ax2.imshow(spectrogram.T, aspect='auto', origin='lower', 
           extent=[times.min(), times.max(), freqs.min(), freqs.max()])
ax2.set_yticks(freqs[::16])
ax2.set_xticks(times[::16])
ax2.set_title('Spectrogram of ' + '0a7c2a8d_nohash_0.wav')
ax2.set_ylabel('Freqs in Hz')
ax2.set_xlabel('Seconds')

samples, sampleRate = librosa.load(audioTrainPath+'yes/0a7c2a8d_nohash_0.wav', sr = 16000)

#resampling the data
ipd.Audio(samples, rate=sampleRate)
print(sampleRate)
samples = librosa.resample(samples, sampleRate, 8000)
ipd.Audio(samples, rate=8000)

#Looking at the number of recordings for a certain word and plotting
labels = os.listdir(audioTrainPath)

numRecordings=[]
for label in labels:
    waves = [f for f in os.listdir(audioTrainPath + '/'+ label) if f.endswith('.wav')]
    numRecordings.append(len(waves))
    
plt.figure(figsize=(30,5))
index = np.arange(len(labels))
plt.bar(index, numRecordings)
plt.xlabel('Words', fontsize=12)
plt.ylabel('Number of recordings', fontsize=12)
plt.xticks(index, labels, fontsize=15, rotation=60)
plt.title('Number of recordings for each word')
plt.show()

labels=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

#looking at the duration of the recordings
duration_of_recordings = []
for label in labels:
    waves = [f for f in os.listdir(audioTrainPath + '/'+ label) if f.endswith('.wav')]
    for wav in waves:
        sampleRate, samples = wavfile.read(audioTrainPath + '/' + label + '/' + wav)
        duration_of_recordings.append(float(len(samples)/sampleRate))
plt.hist(np.array(duration_of_recordings))

#Processing the audio to get ready for training
audioTrainPath = '/Users/teddy/Downloads/tensorflow-speech-recognition-challenge/train/audio/'

allWave = []
allLabel = []
for label in labels:
    print(label)
    waves = [f for f in os.listdir(audioTrainPath + '/'+ label) if f.endswith('.wav')]
    for wav in waves:
        samples, sampleRate = librosa.load(audioTrainPath + '/' + label + '/' + wav, sr = 16000)
        samples = librosa.resample(samples, sampleRate, 8000)
        if(len(samples) == 8000) : 
            allWave.append(samples)
            allLabel.append(label)

le = LabelEncoder()
y = le.fit_transform(allLabel)
classes = list(le.classes_)

y = np_utils.to_categorical(y, num_classes=len(labels))

#reshaping the 2D array to a 3D array
allWave = np.array(allWave).reshape(-1,8000,1)

#Creating the train and test sets
xTrain, xTest, yTrain, yTest = train_test_split(np.array(allWave),np.array(y),stratify=y,test_size = 0.2,random_state=777,shuffle=True)
K.clear_session()
inputs = Input(shape=(8000,1))

activationFunction = 'relu'
optimizerFunction = 'Nadam'
num_classes = len(np.unique(yTrain))
init = 'he_normal'

#Convulutional 1D layers
cnnModel = Conv1D(8,13, padding = 'valid', activation = activationFunction, strides=1)(inputs)
cnnModel = MaxPooling1D(3)(cnnModel)
cnnModel = BatchNormalization()(cnnModel)
cnnModel = Dropout(0.4)(cnnModel)

cnnModel = Conv1D(16, 11, padding = 'valid', activation = activationFunction, strides=1)(cnnModel)
cnnModel = MaxPooling1D(3)(cnnModel)
cnnModel = BatchNormalization()(cnnModel)
cnnModel = Dropout(0.4)(cnnModel)

cnnModel = Conv1D(32, 9, padding = 'valid', activation = activationFunction, strides=1)(cnnModel)
cnnModel = MaxPooling1D(3)(cnnModel)
cnnModel = BatchNormalization()(cnnModel)
cnnModel = Dropout(0.4)(cnnModel)

cnnModel = Conv1D(64, 7, padding = 'valid', activation = activationFunction, strides=1)(cnnModel)
cnnModel = MaxPooling1D(3)(cnnModel)
cnnModel = BatchNormalization()(cnnModel)
cnnModel = Dropout(0.4)(cnnModel)

cnnModel = Conv1D(128, 5, padding = 'valid', activation = activationFunction, strides=1)(cnnModel)
cnnModel = MaxPooling1D(3)(cnnModel)
cnnModel = BatchNormalization()(cnnModel)
cnnModel = Dropout(0.4)(cnnModel)

#Flatten cnnModel
cnnModel = Flatten()(cnnModel)

#Dense Layers
cnnModel = Dense(1000, activation = activationFunction)(cnnModel)
cnnModel = BatchNormalization()(cnnModel)
cnnModel = Dropout(0.3)(cnnModel)

cnnModel = Dense(800, activation = activationFunction)(cnnModel)
cnnModel = BatchNormalization()(cnnModel)
cnnModel = Dropout(0.3)(cnnModel)

cnnModel = Dense(600, activation = activationFunction)(cnnModel)
cnnModel = BatchNormalization()(cnnModel)
cnnModel = Dropout(0.3)(cnnModel)


outputs = Dense(len(labels), activation = 'softmax')(cnnModel)
model = Model(inputs, outputs)
model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = optimizerFunction, metrics = ['accuracy'])

#EarlyStopping stops the model from training once the model stops imroving
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 10, min_delta = 0.0001) 

#ModelCheckpoint saves each model after each epoch
mc = ModelCheckpoint('/Users/teddy/Desktop/stuff/ATCS_y4_-teddyliang/bestModel3.hdf5', monitor = 'val_accuracy', verbose = 1, save_best_only = True, mode = 'max')


history = model.fit(xTrain, yTrain, validation_data = (xTest, yTest), epochs = 100, batch_size = 32, callbacks = [es,mc])

pyplot.plot(history.history['loss'], label='train') 
pyplot.plot(history.history['val_loss'], label='test') 
pyplot.ylim(top=1000)
pyplot.legend()
pyplot.show()

#load the best model
model = load_model('/Users/teddy/Desktop/stuff/ATCS_y4_-teddyliang/bestModel3.hdf5')

#predicts the text when audio is passed in
def predict(audio):
    prob = model.predict(audio.reshape(1,8000,1))
    index = np.argmax(prob[0])
    return classes[index]

modelOutputs = model.predict(xTest)

print(modelOutputs)

score = model.evaluate(xTest, yTest, verbose=2)
print('\nTest accuracy:', score[1])
