#   Matthew Kusman
#   SER Project
#   12/7/2021
#   Derived from DataFlair's implementation found at:
#   https://data-flair.training/blogs/python-mini-project-speech-emotion-recognition/

from genericpath import isdir
import time
import librosa
from numpy.core.numeric import full
import soundfile
import os, glob, pickle, sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from pydub import AudioSegment

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
size_read = 0
observed_emotions=['calm', 'happy', 'fearful', 'disgust']
skipped_files = []

    # **********************Approach**************************
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
    full_dir = data_dir + '/Actor_*/*.wav'
    for filename in glob.glob(full_dir):
        print('Loading file:  %s' % filename)

        size = os.path.getsize(filename)
        print('Size: %.2f' % size)
        global size_read
        size_read += size

        feature = extract_feature(filename, True, True, True)

        if not isinstance(feature, type(None)):
            emotion=emotions[filename.split("-")[2]]
            if emotion not in observed_emotions:
                continue
            
            x.append(feature)
            y.append(filename.split('-')[2])
        else:
            print('%s not mono audio, skipping' % filename)
            global skipped_files
            skipped_files.append(filename)
            size_read -= size

    return x, y

# Extracts the 3 key features from audio file
def extract_feature(file_name, mfcc, chroma, mel):
    # test = AudioSegment.from_wav(file_name)
    # test.set_channels(1)
    # test.export(file_name)
    with soundfile.SoundFile(file_name) as sound_file:
        if sound_file.channels > 1:
            return None

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

# Start timer
start_time = time.time()

# If path specified
if len(sys.argv) > 1:
    if os.path.isdir(sys.argv[1]):
        data_dir = sys.argv[1]
    else:
        print('%s is invalid directory' % sys.argv[1])
        sys.exit()

x,y = load_data()
data = train_test_split(x, y, test_size=.25)
xtrain, xtest, ytrain, ytest = data

print('\nCreating classifier...')
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
model.fit(xtrain, ytrain)
print('\nPredicting using fit model...')
predicted = model.predict(xtest)
accuracy = accuracy_score(ytest, predicted)

print('\nAccuracy: {:.2f}%'.format(round(accuracy*100, 2)))

# End time, calculate duration
end_time = time.time()
time_elapsed = time.strftime('%H:%M:%S', time.gmtime(end_time-start_time))

print('\nRead {:.2f} MB in {:s}'.format(size_read/1048576, time_elapsed))
print('Skipped {:d} files'.format(len(skipped_files)))