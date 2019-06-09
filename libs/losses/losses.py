import tensorflow as tf


def masked_cross_entropy(losses, preds, targets, weight):
    '''
    
    '''
    weighted_mask = tf.where(tf.equal(preds, targets), x=tf.scalar_mul(weight, tf.ones_like(preds, dtype=tf.float32)), y=tf.ones_like(preds, dtype=tf.float32))
    masked_losses =  losses * weighted_mask
    return tf.reduce_sum(masked_losses)