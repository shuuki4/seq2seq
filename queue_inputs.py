import tensorflow as tf
import threading
from collections import OrderedDict


def enqueue_func(sess, queue, placeholders,
                 dataset, coord, num_epoch, enqueue_size=32):

    with tf.device("/cpu:0"):
        enqueue_op = queue.enqueue_many([ph for _, ph in placeholders.items()])

    for _ in range(num_epoch):
        for step, data_dict in enumerate(dataset.train_datas(enqueue_size)):
            feed_dict = {placeholders[k]: v for k, v in data_dict.items()}
            sess.run(enqueue_op, feed_dict=feed_dict)

    coord.request_stop()


def enqueue(sess, queue, dataset, coord, num_epoch):
    """
    Make enqueueing thread that generates data and put it into given queue
    :param sess: tensorflow session
    :param queue: tensorflow queue to put data
    :param dataset: dataset instance
    :param coord: tensorflow coordinator to manage thread
    :param num_epoch: number of epoches to proceed
    :return: enqueueing thread
    """
    placeholders = OrderedDict()
    placeholders['encoder_inputs'] = tf.placeholder(
        shape=(None, None),  # batch_size, max_time
        dtype=tf.int32,
        name='encoder_inputs'
    )
    placeholders['encoder_lengths'] = tf.placeholder(
        shape=(None,),
        dtype=tf.int32,
        name='encoder_lengths'
    )
    placeholders['decoder_inputs'] = tf.placeholder(
        shape=(None, None),  # batch_size, max_time
        dtype=tf.int32,
        name='decoder_inputs'
    )
    placeholders['decoder_lengths'] = tf.placeholder(
        shape=(None,),
        dtype=tf.int32,
        name='decoder_lengths'
    )

    t = threading.Thread(target=enqueue_func,
                         args=[sess, queue, placeholders, dataset, coord, num_epoch])
    return t

