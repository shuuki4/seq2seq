import tensorflow as tf
import numpy as np
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.seq2seq import AttentionWrapper, AttentionWrapperState, \
                                       BahdanauAttention, LuongAttention, \
                                       BasicDecoder, BeamSearchDecoder, dynamic_decode, \
                                       TrainingHelper, sequence_loss


class Seq2SeqModel:

    PAD = 0
    EOS = 1

    def __init__(self, config):
        self.config = config
        self._inputs = {}

        self._build_graph()

    def _build_graph(self):
        inputs = self._build_placeholder()
        encoder_inputs = inputs['encoder_inputs']
        encoder_lengths = inputs['encoder_lengths']
        decoder_inputs = inputs['decoder_inputs']
        decoder_lengths = inputs['decoder_lengths']

        encoder_outputs, encoder_state = self._build_encoder(encoder_inputs, encoder_lengths)
        decoder_outputs, decoder_result_ids, decoder_state, decoder_sequence_lengths = \
            self._build_decoder(encoder_outputs, encoder_state, encoder_lengths,
                                decoder_inputs, decoder_lengths, self.config['training']['is_training'])
        self.decoder_outputs = decoder_outputs
        self.decoder_result_ids = decoder_result_ids

        seq_loss = self._build_loss(decoder_outputs, decoder_inputs, decoder_lengths)
        train_step = self._build_train_step(seq_loss)
        self.loss = seq_loss
        self.train_op = train_step

    @property
    def inputs(self):
        return self._inputs

    def make_feed_dict(self, data_dict):
        feed_dict = {}
        for key in data_dict.keys():
            try:
                feed_dict[self._inputs[key]] = data_dict[key]
            except KeyError:
                raise ValueError('Unexpected argument in input dictionary!')
        return feed_dict

    def _build_placeholder(self):
        self._inputs['encoder_inputs'] = tf.placeholder(
            shape=(None, None), # batch_size, max_time
            dtype=tf.int32,
            name='encoder_inputs'
        )
        self._inputs['encoder_lengths'] = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_lengths'
        )
        self._inputs['decoder_inputs'] = tf.placeholder(
            shape=(None, None), # batch_size, max_time
            dtype=tf.int32,
            name='decoder_inputs'
        )
        self._inputs['decoder_lengths'] = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='decoder_lengths'
        )

        return self._inputs

    def _build_encoder(self, encoder_inputs, encoder_lengths):

        with tf.variable_scope('word_embedding'):
            word_embedding = tf.get_variable(
                name="word_embedding",
                shape=(self.config['word']['num_word'], self.config['word']['embedding_dim']),
                initializer=xavier_initializer(),
                dtype=tf.float32
            )

            # batch_size, max_time, embed_dims
            encoder_input_vectors = tf.nn.embedding_lookup(word_embedding, encoder_inputs)

        with tf.variable_scope('encoder'):
            # build cell
            if self.config['encoder']['cell_type'] == 'LSTM':
                encoder_cell = LSTMCell(self.config['encoder']['state_size'])
            elif self.config['encoder']['cell_type'] == 'GRU':
                encoder_cell = GRUCell(self.config['encoder']['state_size'])
            else:
                raise ValueError

            if self.config['encoder']['bidirection']:
                (fw_output, bw_output), (fw_final_state, bw_final_state) =\
                    tf.nn.bidirectional_dynamic_rnn(
                        encoder_cell, encoder_cell,
                        encoder_input_vectors,
                        sequence_length=encoder_lengths,
                        time_major=False,
                        dtype=tf.float32
                    )

                encoder_outputs = tf.concat([fw_output, bw_output], 2)
                if isinstance(fw_final_state, LSTMStateTuple):
                    encoder_state_c = tf.concat(
                        [fw_final_state.c, bw_final_state.c], 1)
                    encoder_state_h = tf.concat(
                        [fw_final_state.h, bw_final_state.h], 1)
                    encoder_final_state = LSTMStateTuple(encoder_state_c,
                                                         encoder_state_h)
                else:
                    encoder_final_state = tf.concat(
                        [fw_final_state, bw_final_state], 1)

            else:
                encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                    encoder_cell,
                    encoder_input_vectors,
                    sequence_length=encoder_lengths,
                    time_major=False,
                    dtype=tf.float32
                )

            return encoder_outputs, encoder_final_state

    def _build_decoder(self, encoder_outputs, encoder_final_state, encoder_lengths,
                       decoder_inputs, decoder_lengths, is_training):

        batch_size, decoder_time = tf.unstack(tf.shape(decoder_inputs))

        with tf.variable_scope('decoder_cell'):
            state_size = self.config['decoder']['state_size']
            if self.config['encoder']['bidirection']:
                state_size *= 2

            if self.config['decoder']['cell_type'] == 'LSTM':
                decoder_cell = LSTMCell(state_size)
            elif self.config['decoder']['cell_type'] == 'GRU':
                decoder_cell = GRUCell(state_size)
            else:
                raise ValueError

            if self.config['attention']['attention']:
                attention_name = self.config['attention']['attention_type']
                if attention_name == 'Bahdanau':
                    attention_fn = BahdanauAttention
                elif attention_name == 'Luong':
                    attention_fn = LuongAttention
                else:
                    raise ValueError

                attention_mechanism = attention_fn(
                    self.config['attention']['attention_num_units'],
                    encoder_outputs,
                    encoder_lengths
                )
                decoder_cell = AttentionWrapper(
                    decoder_cell,
                    attention_mechanism,
                    attention_layer_size=self.config['attention']['attention_depth'],
                    output_attention=True,
                )

                decoder_initial_state = \
                    decoder_cell.zero_state(batch_size, tf.float32).clone(
                        cell_state=encoder_final_state
                    )

            else:
                decoder_initial_state = encoder_final_state

        with tf.variable_scope('word_embedding', reuse=True):
            word_embedding = tf.get_variable(name="word_embedding")

        with tf.variable_scope('decoder'):
            out_func = layers_core.Dense(
                self.config['word']['num_word'], use_bias=False)

            if is_training:
                eoses = tf.ones([batch_size, 1], dtype=tf.int32) * self.EOS
                eosed_decoder_inputs = tf.concat([eoses, decoder_inputs], 1)
                embed_decoder_inputs = tf.nn.embedding_lookup(word_embedding, eosed_decoder_inputs)

                training_helper = TrainingHelper(
                    embed_decoder_inputs,
                    decoder_lengths + 1
                )
                decoder = BasicDecoder(
                    decoder_cell,
                    training_helper,
                    decoder_initial_state,
                    output_layer=out_func
                )

                decoder_outputs, decoder_state, decoder_sequence_outputs = \
                    dynamic_decode(
                        decoder,
                        # maximum_iterations=self.config['decoder']['max_iteration'] / TrainingHelper stops decode
                    )
                decoder_outputs, decoder_sample_ids = decoder_outputs

            else:
                pass

        return decoder_outputs, decoder_sample_ids, decoder_state, decoder_sequence_outputs

    def _build_loss(self, decoder_logits, decoder_inputs, decoder_lengths):
        with tf.variable_scope('loss_target'):
            #build decoder output, with appropriate padding and mask
            max_decoder_time = tf.reduce_max(decoder_lengths) + 1
            decoder_target = decoder_inputs[:, :max_decoder_time]

            decoder_eos = tf.one_hot(decoder_lengths, depth=max_decoder_time,
                                     on_value=self.EOS, off_value=self.PAD,
                                     dtype=tf.int32)
            decoder_target += decoder_eos

            decoder_loss_mask = tf.sequence_mask(decoder_lengths + 1,
                                                 maxlen=max_decoder_time,
                                                 dtype=tf.float32)

        with tf.variable_scope('loss'):
            seq_loss = sequence_loss(
                decoder_logits,
                decoder_target,
                decoder_loss_mask,
                name='sequence_loss'
            )

        return seq_loss

    def _build_train_step(self, loss):
        with tf.variable_scope('train'):
            lr = self.config['training']['learning_rate']
            opt = tf.train.AdamOptimizer(learning_rate=lr)

            train_variables = tf.trainable_variables()
            grads_vars = opt.compute_gradients(loss, train_variables)
            for i, (grad, var) in enumerate(grads_vars):
                grads_vars[i] = (tf.clip_by_norm(grad, 1.0), var)

            apply_gradient_op = opt.apply_gradients(grads_vars)
            with tf.control_dependencies([apply_gradient_op]):
                train_op = tf.no_op(name='train_step')

        return train_op
