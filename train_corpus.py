import tensorflow as tf
import os
from math import ceil, floor

from model import Seq2SeqModel
from config import Config
from data.corpus_data import CorpusData
from util import log


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('corpus_path', '', """Path of corpus""")
tf.app.flags.DEFINE_string('vocab_path', '', """Path of vocab""")
tf.app.flags.DEFINE_string('train_dir', '', """Directory for train/save""")


def interpret_result(input_ids, output_ids, infer_output_ids, dataset, show=3):
    for i in range(show):
        input_sequence = ' '.join(dataset.interpret(input_ids[i], join_string=' ')
                                  .replace(dataset.symbols[0], '').split())
        output_sequence = ' '.join(dataset.interpret(output_ids[i], join_string=' ')
                                   .replace(dataset.symbols[0], '').split())
        infer_output_sequence = dataset.interpret(infer_output_ids[i], join_string=' ')

        # temporary for calculation seq data
        print('{} -> {} (Real: {})'.format(input_sequence, infer_output_sequence, output_sequence))


def main(argv=None):
    corpus_path = FLAGS.corpus_path
    vocab_path = FLAGS.vocab_path
    train_dir = FLAGS.train_dir

    dataset = CorpusData(max_length=10)
    dataset.load(corpus_path=corpus_path, vocab_path=vocab_path)

    config = Config(is_training=True, num_words=dataset.num_symbols,
                    word_embedding_dim=100, rnn_state_size=200,
                    max_iteration=dataset.max_length + 1,
                    batch_size=128)

    max_epoch = 100
    batch_size = config['training']['batch_size']
    steps_in_epoch = int(floor(dataset.num_train_examples / batch_size))

    model = Seq2SeqModel(config, input_batch=None)
    saver = tf.train.Saver(tf.global_variables())

    summary_step = 1000
    log_step = 3000
    save_step = 50000

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

        log.warning("Training Start!")
        for epoch in range(1, max_epoch+1):
            log.warning("Epoch {}".format(epoch))
            for step, data_dict in enumerate(dataset.train_datas(batch_size)):
                feed_dict = model.make_feed_dict(data_dict)
                run_dict = {'train_op': model.train_op,
                            'decoder_result_ids': model.decoder_result_ids,
                            'loss': model.loss,
                            'step': model.train_step}
                if (step + 1) % summary_step == 0:
                    run_dict['summary_op'] = model.summary_op
                run_results = sess.run(run_dict, feed_dict)

                global_step = run_results['step']
                if (step + 1) % log_step == 0:
                    log.info("Step {cur_step:6d} / {all_step:6d} ... Loss: {loss:.5f}"
                             .format(cur_step=step+1,
                                     all_step=steps_in_epoch,
                                     loss=run_results['loss']))

                    interpret_result(data_dict['encoder_inputs'],
                                     data_dict['decoder_inputs'],
                                     run_results['decoder_result_ids'],
                                     dataset)
                if (step + 1) % summary_step == 0:
                    summary_writer.add_summary(run_results['summary_op'], global_step)
                    summary_writer.flush()

                #save per 50000 step
                if (step + 1) % save_step == 0:
                    log.warning("Saving Checkpoints...")
                    ckpt_path = os.path.join(train_dir, 'model.ckpt')
                    saver.save(sess, ckpt_path, global_step=global_step)
                    log.warning("Saved checkpoint into {}!".format(ckpt_path))


if __name__ == '__main__':
    tf.app.run()
