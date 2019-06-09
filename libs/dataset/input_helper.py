import tensorflow as tf
import numpy as np

class Mnist(object):
    '''
    '''
    def __init__(self, dataset="train"):
        
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        images = x_train if dataset=="train" else x_test
        labels = y_train if dataset=="train" else y_test

        # Preprocessing
        images = images.astype('float32')
        images /= 255
        images = np.expand_dims(images, axis=-1)

        self.images = images
        self.labels = labels
        self.num_samples = len(self.images)
        self.shuffle_samples()
        self.next_batch_pointer = 0

    def shuffle_samples(self):
        image_indices = np.random.permutation(np.arange(self.num_samples))
        self.images = self.images[image_indices]
        self.labels = self.labels[image_indices]

    def get_next_batch(self, batch_size=100):
        num_samples_left = self.num_samples - self.next_batch_pointer
        if num_samples_left >= batch_size:
            batch1 = self.images[self.next_batch_pointer:self.next_batch_pointer + batch_size]
            batch2 = self.labels[self.next_batch_pointer:self.next_batch_pointer + batch_size]

            self.next_batch_pointer += batch_size
        else:
            partial_batch_1 = self.images[self.next_batch_pointer:self.num_samples]
            partial_batch_2 = self.labels[self.next_batch_pointer:self.num_samples]
            self.shuffle_samples()
            partial_batch_11 = self.images[0:batch_size - num_samples_left]
            partial_batch_22 = self.labels[0:batch_size - num_samples_left]

            batch1 = np.concatenate((partial_batch_1, partial_batch_11))
            batch2 = np.concatenate((partial_batch_2, partial_batch_22))
            self.next_batch_pointer = batch_size - num_samples_left
        return batch1, np.eye(10)[batch2]
    
    
    
class Cifar10(object):
    '''
    '''
    def __init__(self, dataset="train"):
        
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        images = x_train if dataset=="train" else x_test
        labels = y_train if dataset=="train" else y_test
        self.images = (images - 127.5) / 127.5
        self.labels = np.squeeze(labels)
        self.num_samples = len(self.images)
        self.shuffle_samples()
        self.next_batch_pointer = 0

    def shuffle_samples(self):
        image_indices = np.random.permutation(np.arange(self.num_samples))
        self.images = self.images[image_indices]
        self.labels = self.labels[image_indices]

    def get_next_batch(self, batch_size=100):
        num_samples_left = self.num_samples - self.next_batch_pointer
        if num_samples_left >= batch_size:
            batch1 = self.images[self.next_batch_pointer:self.next_batch_pointer + batch_size]
            batch2 = self.labels[self.next_batch_pointer:self.next_batch_pointer + batch_size]

            self.next_batch_pointer += batch_size
        else:
            partial_batch_1 = self.images[self.next_batch_pointer:self.num_samples]
            partial_batch_2 = self.labels[self.next_batch_pointer:self.num_samples]
            self.shuffle_samples()
            partial_batch_11 = self.images[0:batch_size - num_samples_left]
            partial_batch_22 = self.labels[0:batch_size - num_samples_left]

            batch1 = np.concatenate((partial_batch_1, partial_batch_11))
            batch2 = np.concatenate((partial_batch_2, partial_batch_22))
            self.next_batch_pointer = batch_size - num_samples_left
        return batch1, np.eye(10)[batch2]

