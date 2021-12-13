## Speech Emotion Recognition (SER)**

**SER Through MLPClassifier and RAVDESS Dataset**
This Project uses sci-kit.learn's Multi-Layer Perceptron Classifier to learn speech recognition from the RAVDESS dataset publicly available. More information can be found in the included report.

**Requirements**
To run the learner, you will need the following libraries installed:

 - argparse
 - librosa
 - soundfile
 - numpy
 - sklearn

**Instructions**
To run the preset parameters, simply run processor.py with python: python processor.py or python ./processor.py

This will run the processor.py script which internally calls the data_processor.py script with the preset parameters and outputs the data to the output.txt file. This will overwrite the current file.

You can also run the data_processor.py script on its own, which has its own defaults but accepts several parameters.

**Defaults:**
Use all 8 emotions
Use dataset within the repo
Run the script on each of the following:

 - alpha = .0001, .001, .01, 1
 - learning rate= adaptive, constant, invscaling


**data_processor.py parameters**
 - -p {path}: path to source data directory in RAVDESS format
 - -e [emotions] OR {a}: list emotions to use, or the 'a' tag to use all present in data: default is the following emotions:  ['calm', 'happy', 'fearful', 'disgust']
 - -v: Verbose updates to std out
 - -P: **WIP**
 - -Lr {learning rate}: must be adaptive, constant, or invscaling. Default is adaptive.
 - -a {float}: alpha value as float
