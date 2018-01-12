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


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class image_object:
    def __init__(self):
        self.image = tf.Variable([], dtype=tf.string)
        self.height = tf.Variable([], dtype=tf.int64)
        self.width = tf.Variable([], dtype=tf.int64)
        self.filename = tf.Variable([], dtype=tf.string)
        self.label = tf.Variable([], dtype=tf.int32)


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features={
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/filename": tf.FixedLenFeature([], tf.string),
        "image/class/label": tf.FixedLenFeature([], tf.int64), })

    image_encoded = features["image/encoded"]
    image_raw = tf.image.decode_jpeg(image_encoded, channels=1)

    current_image_object = image_object()

    current_image_object.image = tf.image.resize_image_with_crop_or_pad(image_raw, 20,
                                                                        20)  # cropped image with size 299x299
    #    current_image_object.image = tf.cast(image_crop, tf.float32) * (1./255) - 0.5
    current_image_object.height = features["image/height"]  # height of the raw image
    current_image_object.width = features["image/width"]  # width of the raw image
    current_image_object.filename = features["image/filename"]  # filename of the raw image
    current_image_object.label = tf.cast(features["image/class/label"], tf.int32)  # label of the raw image

    print("End of readndec")
    return current_image_object


filename_queue = tf.train.string_input_producer(
    ["tfrecords/train-00000-of-00001"],
    shuffle=True)


def process_image(image):
    print("Before eval")
    image = np.reshape(image, (400,))
    image = tf.cast(image, tf.float64) * (1. / 255) - 0.5
    image = image.eval()
    return image.eval()


def next_batch(filenames, imagenumber):
    print("Start next batch")
    with tf.Session() as sess:
        tf.train.start_queue_runners(sess)
        current_image_object = read_and_decode(tf.train.string_input_producer(filenames, shuffle=True))
        images = np.empty((0, 400))
        labels = np.empty((0, 10))
        print("Image", current_image_object.label.eval())
        for i in range(imagenumber):
            print("Loop no: ", i)
            pre_image, pre_label = sess.run([current_image_object.image, current_image_object.label])
            print("after sess run")
            image = process_image(pre_image)
            label_coded = dict[pre_label]
            images = np.concatenate((images, [image]))
            labels = np.concatenate((labels, [label_coded]))
            print("Labels: ", labels)

    return images, labels


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    print("Läuft")
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    epochsnr = 10
    images, labels = next_batch(filenames=["tfrecords/train-00000-of-00001"], imagenumber=173)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochsnr):
            print("Epoch loop")
            epoch_loss = 0
            images, labels = next_batch(filenames=["tfrecords/train-00000-of-00001"], imagenumber=173)
            # input("Press Enter to continue...")
            _, c = sess.run([optimizer, cost], feed_dict={x: images, y: labels})
            epoch_loss += c
            print('Epoch', epoch, 'completed out of', epochsnr, 'loss: ', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        test_x, test_y = next_batch(filenames=["tfrecords/validation-00000-of-00001"], imagenumber=12)
        print('Accuracy', accuracy.eval({x: test_x, y: test_y}))


train_neural_network(x)
