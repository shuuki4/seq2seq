import tensorflow as tf
from math import ceil, floor

from model import Seq2SeqModel
from queue_inputs import enqueue
from config import Config
from data.calculation_data import CalculationSeqData
from util import log


def interpret_result(input_ids, output_ids, dataset, show=3):
    for i in range(show):
        input_sequence = dataset.interpret(input_ids[i]).replace(dataset.symbols[0], '')
        output_sequence = dataset.interpret(output_ids[i])

        # temporary for calculation seq data
        print('{} -> {} (Real: {})'.format(input_sequence, output_sequence, eval(input_sequence)))


def eval_result(input_ids, output_ids, dataset):
    _right, _wrong = 0.0, 0.0
    for i in range(len(input_ids)):
        input_sequence = dataset.interpret(input_ids[i]).replace(dataset.symbols[0], '')
        output_sequence = dataset.interpret(output_ids[i])

        try:
            if eval(input_sequence) == int(output_sequence):
                _right += 1.0
            else:
                _wrong += 1.0
        except ValueError:  # output_sequence == ''
            _wrong += 1.0

    return _right, _wrong


if __name__ == '__main__':

    dataset = CalculationSeqData()
    dataset.build()

    config = Config(is_training=True, num_words=dataset.num_symbols,
                    word_embedding_dim=30, rnn_state_size=100,
                    max_iteration=dataset.max_length + 1,
                    batch_size=128)

    max_epoch = 100
    batch_size = config['training']['batch_size']
    steps_in_epoch = int(floor(dataset.num_train_examples / batch_size))

    """
    with tf.device("/cpu:0"):
        input_queue = tf.FIFOQueue(
            capacity=5 * batch_size,
            dtypes=[tf.int32, tf.int32, tf.int32, tf.int32],
            shapes=[[dataset.max_length], [], [dataset.max_length], []]
        )
        input_batch = tf.train.shuffle_batch(
            input_queue.dequeue(),
            batch_size=batch_size,
            num_threads=1,
            capacity=20 * batch_size,
            min_after_dequeue=4 * batch_size
        )
    """

    model = Seq2SeqModel(config, input_batch=None)

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        log.warning("Training Start!")
        for epoch in range(1, max_epoch+1):
            log.warning("Epoch {}".format(epoch))
            for step, data_dict in enumerate(dataset.train_datas(batch_size)):
                feed_dict = model.make_feed_dict(data_dict)
                _, decoder_result_ids, loss_value = \
                    sess.run([model.train_op, model.decoder_result_ids, model.loss], feed_dict)

                if (step + 1) % 300 == 0:
                    log.info("Step {cur_step:6d} / {all_step:6d} ... Loss: {loss:.5f}"
                             .format(cur_step=step+1,
                                     all_step=steps_in_epoch,
                                     loss=loss_value))

                    interpret_result(data_dict['encoder_inputs'], decoder_result_ids, dataset)

            right, wrong = 0.0, 0.0
            for step, data_dict in enumerate(dataset.val_datas(batch_size)):
                feed_dict = model.make_feed_dict(data_dict)
                beam_result_ids = sess.run(model.beam_search_result_ids, feed_dict)[:, :, 0]
                if step == 0:
                    print(data_dict['decoder_inputs'][:5])
                    print(beam_result_ids[:5])
                now_right, now_wrong = eval_result(data_dict['encoder_inputs'], beam_result_ids, dataset)
                right += now_right
                wrong += now_wrong

            log.infov("Right: {}, Wrong: {}, Accuracy: {:.2}%".format(right, wrong, 100*right/(right+wrong)))

        """
        coord = tf.train.Coordinator()

        input_thread = enqueue(sess, input_queue, dataset, coord, max_epoch)
        input_thread.start()
        tf.train.start_queue_runners(sess=sess, coord=coord)

        step = 0
        log.warning("Training Start!")
        try:
            while not coord.should_stop():
                if step % steps_in_epoch == 0:
                    log.warning("Epoch {}".format(int(step / steps_in_epoch) + 1))

                _, encoder_inputs, decoder_result_ids, loss_value = \
                    sess.run([model.train_op, model.encoder_inputs,
                              model.decoder_result_ids, model.loss])

                if (step + 1) % 300 == 0:
                    log.info("Step {cur_step:6d} / {all_step:6d} ... Loss: {loss:.5f}"
                             .format(cur_step=(step+1) % steps_in_epoch,
                                     all_step=steps_in_epoch,
                                     loss=loss_value))
                    interpret_result(encoder_inputs, decoder_result_ids, dataset)

                step += 1

        except tf.errors.OutOfRangeError:
            print("Done training!")
        finally:
            coord.request_stop()

        coord.join([input_thread])
        """
