import argparse
import operator

import cv2
import numpy as np
import tensorflow as tf


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
        return graph


def outputTensorToLabel(tensor):
    index = np.argmax(tensor)
    if (index < 9):
        return str((index+1) * 10) + "Km/h"
    else:
        return "No sign"

if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="modelsnewIII/frozen_model.pb", type=str,
                        help="Frozen model file to import")
    parser.add_argument("--image", default="images/cropped/10/10.jpg", type=str,
                        help="Path to image")
    args = parser.parse_args()
    graph = load_graph(args.frozen_model_filename)
    image_path = args.image
    # We can verify that we can access the list of operations in the graph
    # for op in graph.get_operations():
        # print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions

    # We access the input and output nodes
    x = graph.get_tensor_by_name('prefix/input:0')
    y = graph.get_tensor_by_name('prefix/Add_2:0')

    # We launch a Session
    with tf.Session(graph=graph) as sess:
        #Load specified image
        im = cv2.imread(image_path, 0)
        ten = np.reshape(im, (400,))
        ten1 = ten / 255 - 0.5
        #run actual inference
        y_out = sess.run(y, feed_dict={
            x: [ten1]
        })
        print(y_out)
        print(outputTensorToLabel(y_out))
