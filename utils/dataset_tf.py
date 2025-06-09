import h5py
import os
import tensorflow as tf


def generator_sti(path):
    files = os.listdir(path)
    for file in files:
        filename = os.path.join(path, file)
        with h5py.File(filename, 'r') as hf:
            phase = hf.get('phase')
            phase = tf.convert_to_tensor(phase)
            chi = hf.get('chi')
            chi = tf.convert_to_tensor(chi)
            yield {'phase': 1e6 * phase, 'chi': 1e6 * chi}
