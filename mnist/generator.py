import tensorflow as tf

class Generator:
    
    def __init__(self):
            self.dn1 = tf.layers.Dense(1024)
            self.bn1 = tf.layers.BatchNormalization(name="bn1")
            self.dn2 = tf.layers.Dense(7*7*128)
            self.bn2 = tf.layers.BatchNormalization()

            self.deconv1 = tf.layers.Conv2DTranspose(128, 5, [2,2], padding='same')
            self.bn3 = tf.layers.BatchNormalization()
            self.deconv2 = tf.layers.Conv2DTranspose(1, 5, [2,2], padding='same', activation=tf.nn.sigmoid)

    def __call__(self, z, y, is_training=True):
        with tf.compat.v1.variable_scope("generator"):
            h0 = tf.concat([z, y], axis=-1)

            h1 = self.dn1(h0)
            h1 = self.bn1(h1, training=is_training)
            h1 = tf.nn.relu(h1)

            h2 = tf.concat([h1, y], axis=-1)
            h2 = self.dn2(h2)
            h2 = self.bn2(h2, training=is_training)
            h2 = tf.nn.relu(h2)

            h3 = tf.reshape(h2, [-1, 7,7,128])
            y_expand = tf.tile( tf.reshape(y,[-1,1,1,10]), [1,7,7,1])
            h3 = tf.concat([h3,y_expand], axis=-1)
            h3 = self.deconv1(h3)
            h3 = self.bn3(h3, training=is_training)
            h3 = tf.nn.relu(h3)

            y_expand = tf.tile( tf.reshape(y,[-1,1,1,10]), [1,14,14,1])
            h4 = tf.concat([h3,y_expand], axis=-1)
            h4 = self.deconv2(h4)
            return h4