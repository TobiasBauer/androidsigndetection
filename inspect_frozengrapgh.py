import tensorflow as tf
import argparse


def inspectgraph(name, dir):
    g = tf.GraphDef()

    g.ParseFromString(open(dir + "/frozen_model.pb", "rb").read())

    print([n for n in g.node if n.name.find(name) != -1]) # same for output or any other node you want to make sure is ok

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="", help="Enter node name")
    parser.add_argument("--dir", type=str, default="", help="Enter directory name")

    args = parser.parse_args()
    inspectgraph(args.name, args.dir)