import tensorflow as tf



input_size = 400
n_nodes_hl1 = 100
n_nodes_hl2 = 100

n_classes = 10
batch_size = 100
epochsnr = 10000

input_node_name = 'input'
keep_prob_node_name = 'keep_prob'
output_node_name = 'output_layer'

x = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name=input_node_name)
keep_prob = tf.placeholder(dtype=tf.float32, name=keep_prob_node_name)
y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])

def neural_network_model(data):
    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([input_size, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)

    output = tf.add(tf.matmul(l2, output_layer['weights']), output_layer['biases'])

    return output
# Create some variables.
# Add ops to save and restore all the variables.
tf.reset_default_graph()

neural_network_model(x)
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.


with tf.Session() as sess:
    saver = tf.train.Saver()
    res = saver.restore(sess, 'models/my-model')
    print(res)
    input("Hode")