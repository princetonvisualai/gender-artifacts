import pickle, argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--labels', type=str, default=None)
parser.add_argument('--labels_train', type=str, default=None)
parser.add_argument('--labels_val', type=str, default=None)
parser.add_argument('--labels_test', type=str, default=None)
arg = vars(parser.parse_args())
print('\n', arg, '\n')

# Load the processed labels
labels = pickle.load(open(arg['labels'], 'rb'))

# Do a 60-40 split of train
N = len(list(labels.keys()))
N_60 = int(N*0.6)

# Select images to be added to the 80 split
np.random.seed(1234)
keys = list(labels.keys())
inds_60 = np.random.choice(N, N_60, replace=False)
keys_60 = np.array(keys)[inds_60] # train
keys_40 = np.delete(keys, inds_60)

# split the 40 split into two - val, test
N = len(keys_40)
N_50 = int(N*0.5)
keys = keys_40
inds_50 = np.random.choice(N, N_50, replace=False)
keys_50 = np.array(keys)[inds_50] # test 
keys_40 = np.delete(keys, inds_50) # val


# Create smaller label dictionaries
labels_train = {k: labels[k] for k in keys_60}
labels_val = {k: labels[k] for k in keys_40}
labels_test = {k: labels[k] for k in keys_50}

with open(arg['labels_train'], 'wb') as handle:
    pickle.dump(labels_train, handle, protocol=4)
with open(arg['labels_val'], 'wb') as handle:
    pickle.dump(labels_val, handle, protocol=4)
with open(arg['labels_test'], 'wb') as handle:
    pickle.dump(labels_test, handle, protocol=4)
