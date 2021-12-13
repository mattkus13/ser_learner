# Simply runs the data_processor.py for the specified parameters and formats output

import os

# Parameter arrays to pass to the script
alphas = [.0001, .001, .01, .1]
learning_rates = ['adaptive', 'constant', 'invscaling']

# Clear output.txt
with open('output.txt', 'w') as out:
    out.close()

# Call script for each parameter combination
# -e a: use all emotions found
# -Lr {str}: Learning rate
# -a {float}: alpha value
for rate in learning_rates:
    for alpha in alphas:
        os.system('py .\data_processing.py -e a -Lr %s -a %s' % (rate, str(alpha)))
        with open('output.txt', 'a') as out:
            out.write('*****************************\n')