import tensorflow as tf
from math import ceil

from model import Seq2SeqModel
from config import Config
from data.calculation_data import CalculationSeqData
from util import log


def interpret_result(input_ids, output_ids, symbols, eos_idx, show=3):
    for i in range(show):
        input_idxs = []
        for j in input_ids[i]:
            if j != 0:
                input_idxs.append(j)
            else:
                break

        output_idxs = []
        for j in output_ids[i]:
            if j != eos_idx:
                extra = 1 if j > eos_idx else 0
                output_idxs.append(j - extra)
            else:
                break

        input_sequence = ''.join([symbols[idx] for idx in input_idxs])
        output_sequence = ''.join([symbols[idx] for idx in output_idxs])
        print('{} -> {}'.format(input_sequence, output_sequence))

if __name__ == '__main__':

    ##just train calculation dataset
    dataset = CalculationSeqData()
    dataset.build()

    config = Config(is_training=True, num_words=dataset.num_symbols,
                    word_embedding_dim=30, rnn_state_size=30,
                    max_iteration=dataset.max_length + 1)
    model = Seq2SeqModel(config)

    max_epoch = 100
    batch_size = 16
    all_step = int(ceil(dataset.num_train_examples / batch_size))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        log.warning("Training Start!")
        for epoch in range(1, max_epoch+1):
            log.warning("Epoch {}".format(epoch))
            for step, data_dict in enumerate(dataset.train_datas(batch_size)):
                feed_dict = model.make_feed_dict(data_dict)
                _, decoder_result_ids, loss_value = \
                    sess.run([model.train_op, model.decoder_result_ids, model.loss], feed_dict)

                if (step + 1) % 1000 == 0:
                    log.info("Step {cur_step:6d} / {all_step:6d} ... Loss: {loss:.5f}"
                             .format(cur_step=step+1,
                                     all_step=all_step,
                                     loss=loss_value))

                    interpret_result(data_dict['encoder_inputs'], decoder_result_ids,
                                     dataset.symbols, model.EOS)
