import tensorflow as tf
from math import ceil

from model import Seq2SeqModel
from config import Config
from data.calculation_data import CalculationSeqData
from util import log


if __name__ == '__main__':

    dataset = CalculationSeqData()
    dataset.build()

    config = Config(is_training=True, num_words=dataset.num_symbols,
                    word_embedding_dim=30, rnn_state_size=100,
                    max_iteration=dataset.max_length + 1,
                    batch_size=16, beam_width=1)

    max_epoch = 3
    batch_size = config['training']['batch_size']
    steps_in_epoch = int(ceil(dataset.num_train_examples / batch_size))
    model = Seq2SeqModel(config, input_batch=None)

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        for i in range(1000):
            feed_dict = model.make_feed_dict(dataset.train_data_by_idx(i * batch_size, (i+1) * batch_size))
            sess.run(model.train_op, feed_dict)
        feed_dict = model.make_feed_dict(dataset.val_data_by_idx(2407, 2407+batch_size))
        _, loss, decoder_result_ids, beam_search_result_ids, beam_search_scores = \
            sess.run([model.train_op, model.loss, model.decoder_result_ids,
                      model.beam_search_result_ids, model.beam_search_scores], feed_dict)
        print('Decoder_result_ids', decoder_result_ids.shape)
        print('Beam_search_result_ids', beam_search_result_ids.shape)
        print(decoder_result_ids)
        print(beam_search_result_ids[:, :, 0])
        #print(beam_search_result_ids[:, :, 1])

