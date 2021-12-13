#   Matthew Kusman
#   SER Project
#   12/7/2021
#   Derived from DataFlair's implementation found at:
#   https://data-flair.training/blogs/python-mini-project-speech-emotion-recognition/

import argparse
from genericpath import isdir
import time
import librosa
import soundfile
import os, glob, sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Globals:
data_dir = './ravdess/'
data_dir = ".\\ravdessF"
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
# Set default values
size_read = 0
observed_emotions=list(emotions.values())
observed_emotions_key = list(emotions.keys())
skipped_files = []
args = None
learning_rate = 'adaptive'
alpha = .01

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
    actor = None
    full_dir = data_dir + '/Actor_*/*.wav'
    for filename in glob.glob(full_dir):
        # Update actor for output
        cur_actor = filename.split('Actor')[1][1:3]
        if cur_actor != actor:
            actor = cur_actor
            print('Reading Actor_%s files'%actor)

        if args.v: print('Loading file:  %s' % filename)

        # For debugging and stats, get size and add to total bytes read
        size = os.path.getsize(filename)
        if args.v: print('Size: %.2f' % size)
        global size_read
        size_read += size

        # Extract the features
        feature = extract_feature(filename, True, True, True)

        # extract_features returns none if not mono audio
        # in that case, remove it from size and do not append it
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
        # Extract mfcc
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        # Extract chroma
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        # Extract mel-spectogram
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
        # Return a numpy array of all features
    return result

# Given list of predicted emotions and true emotions, determine stats
# Stats: 
# % emotion prediction is correct
def calculate_stats(predicted_data, true_data):
    # Check same length
    if len(predicted_data) != len(true_data):
        return None
    
    # Create dict from list of emotions used default the counter to 0
    emotion_count = dict.fromkeys(observed_emotions_key, 0)
    # Same for correct counter
    emotion_correct = dict.fromkeys(observed_emotions_key, 0)

    #Go throug every item in true, add counter to emotion, add counter to emotion_right if correct
    for i in range(len(true_data)):
        emotion_count[true_data[i]] += 1
        if true_data[i] == predicted_data[i]:
            emotion_correct[true_data[i]] += 1

    # Calculate percentage of correct for each emotion
    for emotion in emotion_correct.keys():
        emotion_correct[emotion] = round(emotion_correct[emotion]/emotion_count[emotion]*100, 2)

    return emotion_correct


# Handle args
def args_handler():
    # Add an argument
    parser.add_argument('-p', type=str, required=False)
    parser.add_argument('-e', nargs='+', required=False, help='specify emotions. Valid emotions are: \nneutral, calm, happy, sad, angry, fearful, disgust, surprised')
    parser.add_argument('-v', action='store_true', help='include for higher verbosity')
    parser.add_argument('-P', type = str)
    parser.add_argument('-Lr', type=str)
    parser.add_argument('-a', type=float)

    global args
    args = parser.parse_args()

    # If path specified
    if args.p:
        if os.path.isdir(args.p):
            data_dir = args.p
        else:
            print('%s is invalid directory' % sys.argv[1])
            sys.exit()

    # If emotions specified, validate and make observed only those specified
    # Else defaults
    global observed_emotions
    global observed_emotions_key
    if args.e:
        if (len(args.e) == 1) and (args.e[0] == 'a'):
            pass
        else:
            for item in args.e:
                if item not in emotions.values():
                    print('invalid emotion %s'% item)
                    sys.exit()
            for item in observed_emotions:
                if item not in args.e:
                    observed_emotions.remove(item)
    else:
        observed_emotions=['calm', 'happy', 'fearful', 'disgust']
    # Filter the observed emotions key list
    observed_emotions_key = [x for x in observed_emotions_key if emotions[x] in observed_emotions]

    # Update arguments for learning
    global alpha
    global learning_rate
    if args.a: alpha = args.a
    if args.Lr:
        if args.Lr in ['constant', 'invscaling', 'adaptive']:
            learning_rate = args.Lr
        else:
            print('Invalid learning rate')
            sys.exit()

##################################################################################################################
parser = argparse.ArgumentParser()
args_handler()

# Output is what will be written to output file
output = []
output.append('Learning rate: %s\n'%learning_rate)
output.append('alpha: %s\n'%str(alpha))

# Start timer
start_time = time.time()

# Load data, internally calls extract feature for each file
x,y = load_data()
# Split into train/test data
data = train_test_split(x, y, test_size=.25)
xtrain, xtest, ytrain, ytest = data

# Initialize classifier with given parameters plus some hard coded
print('\nCreating classifier...')
model=MLPClassifier(alpha=alpha, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate=learning_rate, max_iter=500)

# Fit model with read data
model.fit(xtrain, ytrain)
print('\nPredicting using fit model...')

# Predict using the fit model, and score using test y
predicted = model.predict(xtest)
accuracy = accuracy_score(ytest, predicted)

print('\nAccuracy: {:.2f}%'.format(round(accuracy*100, 2)))
output.append('Accuracy: {:.2f}%\n'.format(round(accuracy*100, 2)))

# End time, calculate duration
end_time = time.time()
time_elapsed = time.strftime('%H:%M:%S', time.gmtime(end_time-start_time))

print('\nRead {:.2f} MB in {:s}'.format(size_read/1048576, time_elapsed))
if args.v: print('Skipped {:d} files'.format(len(skipped_files)))

# Calculate individual accuracies
print('\nEmotion accuracies:\n')
accuracies = calculate_stats(predicted, ytest)
for emotion in accuracies.keys():
    print('{:s} {:4.2f}%'.format(emotions[emotion], accuracies[emotion]))
    output.append('{:s} {:4.2f}%\n'.format(emotions[emotion], accuracies[emotion]))

with open('output.txt', 'a') as outfile:
    for item in output:
        outfile.write(item)

# WIP
""" if args.P:
    print('Predicting audio file at %s' %args.P)
    data_dir = args.P
    x,y = load_data()
    print(model.predict(x)) """