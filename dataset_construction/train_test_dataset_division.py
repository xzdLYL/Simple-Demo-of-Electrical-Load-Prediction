import numpy as np

data = np.load('../data/input_data.npy')
object = np.load('../data/object_data.npy')
train_data = data[0:20000]
train_label = object[0:20000]
test_data = data[20000:30000]
test_label = object[20000:30000]
np.save('../data/train_input', train_data)
np.save('../data/train_label', train_label)
np.save('../data/test_input', test_data)
np.save('../data/test_label', test_label)
