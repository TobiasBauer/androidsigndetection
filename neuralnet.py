import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("/tmp/data", one_hot=True)


input_size = 400
n_nodes_hl1 = 100
n_nodes_hl2 = 100

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, input_size])
y = tf.placeholder('float')


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


dict = {
    1: [0.] * 0 + [1.] + [0.] * 9,
    2: [0.] * 1 + [1.] + [0.] * 8,
    3: [0.] * 2 + [1.] + [0.] * 7,
    4: [0.] * 3 + [1.] + [0.] * 6,
    5: [0.] * 4 + [1.] + [0.] * 5,
    6: [0.] * 5 + [1.] + [0.] * 4,
    7: [0.] * 6 + [1.] + [0.] * 3,
    8: [0.] * 7 + [1.] + [0.] * 2,
    9: [0.] * 8 + [1.] + [0.] * 1,
    10: [0.] * 9 + [1.] + [0.] * 0

}

'''
def _parse_function(example_proto):
    keys_to_features = {'image/encoded': tf.FixedLenFeature([], tf.string),
                        'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=0),
                        'image/class/text': tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(example_proto, keys_to_features)

    image_raw = parsed_features['image/encoded']
    # Decode the raw bytes so it becomes a tensor with type.
    image = tf.decode_raw(image_raw, tf.uint8)

    # The type is now uint8 but we need it to be float.
    image = tf.cast(image, tf.float32)
    image = image * (1. / 255) - 0.5
    label = parsed_features['image/class/label']

    # label = dict[label_id]

    return image, label
'''


def parse(filename_queue):

    features = \
        {
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/filename': tf.FixedLenFeature([], tf.string)
        }


    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=filename_queue,
                                             features=features)

    # Get the image as raw bytes.
    image_raw = parsed_example['image/encoded']
    # print("Image raw: ", image_raw.eval())
    # Decode the raw bytes so it becomes a tensor with type.
    image = tf.image.decode_jpeg(image_raw, channels=1)

    # image = tf.decode_raw(image_raw, tf.uint8)
    # print("decoded: ", image)
    # The type is now uint8 but we need it to be float.
    image = tf.cast(image, tf.float64) * (1. / 255) - 0.5

    image = image.eval()
    image = np.reshape(image, (400,))

    # Get the label associated with the image.
    label = parsed_example['image/class/label']

    filename = parsed_example['image/filename']
    filename = filename.eval()
    index = label.eval()

    print("Label: ", index)
    print("filename: ", filename)
    input("Enter...")
    label = dict[index]
    # label = dict[index]
    # print("dict[class label]: ", label)
    # input("Enter to continue...")
    # The image and label are now correct TensorFlow types.
    return image, label, filename




def next_batch(filenames, pieces):
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    iterator = dataset.make_one_shot_iterator()

    images = np.empty((0, 400))
    labels = np.empty((0, 10))
    for i in range(pieces):
        elem = iterator.get_next()

        image, label, filename = parse(elem)
        print("image.shape", image.shape)
        print("image type: ", type(image))
        print("images.shape", images.shape)
        print("images type: ", type(images))

        input("Enter")
        images = np.concatenate((images, [image]))
        labels = np.concatenate((labels, [label]))

        # print("filename: ", filename.eval())
    return images, labels


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    epochsnr = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochsnr):
            epoch_loss = 0
            images, labels = next_batch(filenames="tfrecords/train-00000-of-00001", pieces=86)
            # input("Press Enter to continue...")
            _, c = sess.run([optimizer, cost], feed_dict={x: images, y: labels})
            epoch_loss += c
            print('Epoch', epoch, 'completed out of', epochsnr, 'loss: ', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        test_x, test_y = next_batch(filenames=["tfrecords/validation-00000-of-00001"], pieces=12)
        print('Accuracy', accuracy.eval({x: test_x, y: test_y}))

def load_neural_network
train_neural_network(x)
