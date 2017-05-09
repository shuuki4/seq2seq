from model import Seq2SeqModel
from config import Config
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    config = Config(is_training=True, num_words=200)
    model = Seq2SeqModel(config)

    # simple test
    batch_size = 16
    max_time = 10
    input_shape = (batch_size, max_time)

    encoder_inputs = np.random.randint(3, 190, input_shape, dtype=np.int32)
    encoder_lengths = np.ones((batch_size,), dtype=np.int32) * max_time
    decoder_inputs = np.random.randint(3, 190, input_shape, dtype=np.int32)
    decoder_lengths = np.ones((batch_size,), dtype=np.int32) * max_time

    feed_dict = {
        model.inputs['encoder_inputs']: encoder_inputs,
        model.inputs['encoder_lengths']: encoder_lengths,
        model.inputs['decoder_inputs']: decoder_inputs,
        model.inputs['decoder_lengths']: decoder_lengths
    }

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        result = sess.run(model.decoder_outputs, feed_dict=feed_dict)
        print(result)
