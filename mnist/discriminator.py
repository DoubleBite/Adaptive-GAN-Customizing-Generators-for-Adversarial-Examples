import sys
sys.path.append("..")
import tensorflow as tf
from libs.layers.sn_layer import SNConv2D, SNLinear

class Discriminator:
    
    def __init__(self):
            self.conv1 = SNConv2D(10, [5,5], (2,2), padding='same')
            self.conv2 = SNConv2D(64, [5,5], (2,2), padding='same')
            self.dn1 = SNLinear(1024)
            self.dn2 = SNLinear(1)
            self.dn3 = SNLinear(10)
        
    def __call__(self, x, y, is_training=True, reuse=False):
        with tf.compat.v1.variable_scope("discriminator"):
            y_expand = tf.tile( tf.reshape(y,[-1,1,1,10]), [1,28,28,1])
            h0 = tf.concat([x, y_expand], axis=-1)
            h0 = self.conv1(h0)
            h0 = tf.nn.leaky_relu(h0, 0.2)

            y_expand = tf.tile( tf.reshape(y,[-1,1,1,10]), [1,14,14,1])
            h1 = tf.concat([h0, y_expand], axis=-1)
            h1 = self.conv2(h1)
            h1 = tf.nn.leaky_relu(h1, 0.2)

            h2 = tf.layers.flatten(h1)
            h2 = tf.concat([h2, y], axis=-1)
            h2 = self.dn1(h2)
            h2 = tf.nn.leaky_relu(h2, 0.2)

            h3 = self.dn2(h2)
            aux = self.dn3(h2)
            return h3, aux