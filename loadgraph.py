import tensorflow as tf

tf.reset_default_graph()
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

# Create some variables.
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.


with tf.Session() as sess:
    saver = tf.train.Saver()

    saver.restore(sess, 'models/model.ckpt')
    tf.train.write_graph(sess.graph_def,  'models', 'model.pbtxt')
