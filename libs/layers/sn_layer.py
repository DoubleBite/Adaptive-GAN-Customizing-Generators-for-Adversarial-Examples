import tensorflow as tf
from keras.utils import conv_utils



class SNConv2D(tf.keras.layers.Layer):
    '''
    
    '''
    
    def __init__(self, filters, kernel_size, strides=(1,1), padding="same", iterations=1):
        '''
        '''
        self.filters = filters
        self.kernel_size = list(kernel_size)
        self.strides = (1,) + strides + (1,)
        self.padding = padding.upper()
        self.iterations = iterations
        super(SNConv2D, self).__init__()

    def build(self, input_shape):
        '''
        '''
        # Create a trainable weight variable for this layer.
#         print(input_shape, type(input_shape))
#         print(self.kernel_size, type(self.kernel_size))
#         print(self.kernel_size +  [input_shape.as_list()[-1], self.filters])
        self.W = self.add_variable(name='kernel', 
                                      shape= self.kernel_size +  [input_shape.as_list()[-1], self.filters],
                                      initializer=tf.truncated_normal_initializer(stddev=0.02),
                                      trainable=True)
        print(self.W)
        self.bias = self.add_variable(name="bias",
                                      shape=[self.filters],
                                      initializer=tf.constant_initializer(0.0),
                                      trainable=True)
        self.u = self.add_variable(name="u",
                                   shape=[self.W.shape.as_list()[0], 1],
                                   initializer=tf.truncated_normal_initializer(),
                                   trainable=False)
        
        super(SNConv2D, self).build(input_shape)  # Be sure to call this at the end

    def _spectral_normalization(self, W, u):
        '''
        
        '''
        # Reshape W to 2d
        W_reshape = tf.reshape(W, [W.shape.as_list()[0], -1])
        
        # Power iteration method
        u_tilde = u
        v_tilde = None
        
        for _ in range(self.iterations):
            v_tilde = tf.matmul(W_reshape, u_tilde, transpose_a=True) / tf.norm(tf.matmul(W_reshape, u_tilde, transpose_a=True))
            u_tilde = tf.matmul(W_reshape, v_tilde) / tf.norm(tf.matmul(W_reshape, v_tilde))
        sigma = tf.matmul( tf.matmul(u_tilde, W_reshape, transpose_a=True), v_tilde) ## The spectral norm

        # Normalize W and reshape back
        Wsn = W_reshape / sigma
        Wsn = tf.reshape(Wsn, W.shape.as_list())

        # Record the state of u_tilde
        #self.add_update([self.u.assign(u_tilde)])
        self.u.assign(u_tilde)
        
        return Wsn

    
    def call(self, x):
        Wsn = self._spectral_normalization(self.W, self.u)
        conv = tf.nn.conv2d(x, Wsn, strides=self.strides, padding=self.padding)
        return tf.nn.bias_add(conv, self.bias)

    def compute_output_shape(self, input_shape):
            h_old = input_shape[1]
            w_old = input_shape[2]
            h_new = conv_utils.conv_output_length(h_old, self.kernel_size[1], padding=self.padding, stride=self.strides[1])
            w_new = conv_utils.conv_output_length(w_old, self.kernel_size[2], padding=self.padding, stride=self.strides[2])
            return (input_shape[0], h_new, w_new, self.filters)
        
        
        
        
class SNLinear(tf.keras.layers.Layer):
    '''
    
    '''
    
    def __init__(self, output_dim, iterations=1):
        '''
        '''
        self.output_dim = output_dim
        self.iterations = iterations
        super(SNLinear, self).__init__()

    def build(self, input_shape):
        '''
        '''
        # Create a trainable weight variable for this layer.
        self.W = self.add_variable(name='W', 
                                      shape= [input_shape.as_list()[-1], self.output_dim],
                                      initializer=tf.truncated_normal_initializer(stddev=0.02),
                                      trainable=True)
        self.bias = self.add_variable(name="bias",
                                      shape=[self.output_dim],
                                      initializer=tf.constant_initializer(0.0),
                                      trainable=True)
        self.u = self.add_variable(name="u",
                                   shape=[self.W.shape.as_list()[0], 1],
                                   initializer=tf.truncated_normal_initializer(),
                                   trainable=False)
        
        super(SNLinear, self).build(input_shape)  # Be sure to call this at the end

    def _spectral_normalization(self, W, u):
        '''
        
        '''
        # Power iteration method
        u_tilde = u
        v_tilde = None
        
        for _ in range(self.iterations):
            v_tilde = tf.matmul(W, u_tilde, transpose_a=True) / tf.norm(tf.matmul(W, u_tilde, transpose_a=True))
            u_tilde = tf.matmul(W, v_tilde) / tf.norm(tf.matmul(W, v_tilde))
        sigma = tf.matmul( tf.matmul(u_tilde, W, transpose_a=True), v_tilde) ## The spectral norm
        # Normalize W and reshape back
        Wsn = W / sigma

        # Record the state of u_tilde
        #self.add_update([self.u.assign(u_tilde)])
        self.u.assign(u_tilde)
        
        return Wsn

    
    def call(self, x):
        Wsn = self._spectral_normalization(self.W, self.u)
        result = tf.matmul(x, Wsn)
        return tf.nn.bias_add(result, self.bias)

    def compute_output_shape(self, input_shape):
            return (input_shape[0],self.output_dim)