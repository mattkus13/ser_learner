#   Matthew Kusman
#   SER Project
#   12/7/2021
#   Derived from DataFlair's implementation found at:
#   https://data-flair.training/blogs/python-mini-project-speech-emotion-recognition/

import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Globals:
data_dir = './ravdess/'
emotions ={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
} 

    # TODO:
    # Create function to extract mfcc, chroma, and mel features from files
    #   use soundfile to open/read
    #   return result as numpy array
    #   if chroma is true, get short time fourier transformer
    # create dict of all available emotions from RAVDESS
    # load data
    #   takes test %
    #   use glob to get all file names
    #       \\Actor_*\\*.wav
    #       third number is emotion
    #   extract feature (our function)
    #   store feature, label
    # use train_test_split()
    # use MLPClassifier (sklearn) 
    #   fit
    #   predict
    # score: acuracy_score()

# Load data using extract_feature() and glob
def load_data():
    x = []
    y = []
    for filename in glob.glob(data_dir + '/Actor_*/*.wav'):
        y.append(filename.split('-')[2])
        x.append(extract_feature(filename, True, True, True))
    return x, y

# Extracts the 3 key features from audio file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype='float32')
        sample_rate = sound_file.samplerate

        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

x,y = load_data()
data = train_test_split(x, y, test_size=.25)
xtrain, xtest, ytrain, ytest = data
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500, shuffle=True)
model.fit(xtrain, ytrain)
predicted = model.predict(xtest)
accuracy = accuracy_score(ytest, predicted)

print(accuracy)
print('Well that went swimmingly!')