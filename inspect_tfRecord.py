import tensorflow as tf
i = 1
for example in tf.python_io.tf_record_iterator("tfrecords/train-00000-of-00001"):
    result = tf.train.Example.FromString(example)
    print(result)
    print("i = ", i)
    i += 1
