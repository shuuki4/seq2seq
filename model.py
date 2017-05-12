import tensorflow as tf
import numpy as np
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.seq2seq import AttentionWrapper, AttentionWrapperState, \
                                       BasicDecoder, BeamSearchDecoder, dynamic_decode, \
                                       TrainingHelper, sequence_loss, tile_batch

from module.reusable_attention_mechanism import ReusableBahdanauAttention as BahdanauAttention
from module.reusable_attention_mechanism import ReusableLuongAttention as LuongAttention


class Seq2SeqModel:

    PAD = 0
    EOS = 1

    def __init__(self, config, input_batch=None):
        self.config = config
        self._inputs = {}

        self._build_graph(input_batch)

    def _build_graph(self, input_batch):
        encoder_inputs, encoder_lengths, decoder_inputs, decoder_lengths = self._build_inputs(input_batch)
        self.encoder_inputs = encoder_inputs

        encoder_outputs, encoder_state = self._build_encoder(encoder_inputs, encoder_lengths)
        decoder_result = self._build_decoder(encoder_outputs, encoder_state, encoder_lengths,
                                             decoder_inputs, decoder_lengths)
        self.decoder_outputs = decoder_result['decoder_outputs']
        self.decoder_result_ids = decoder_result['decoder_result_ids']
        self.beam_search_result_ids = decoder_result['beam_decoder_result_ids']
        self.beam_search_scores = decoder_result['beam_decoder_scores']

        seq_loss = self._build_loss(self.decoder_outputs, decoder_inputs, decoder_lengths)
        train_step = self._build_train_step(seq_loss)
        self.loss = seq_loss
        self.train_op = train_step

    def make_feed_dict(self, data_dict):
        feed_dict = {}
        for key in data_dict.keys():
            try:
                feed_dict[self._inputs[key]] = data_dict[key]
            except KeyError:
                raise ValueError('Unexpected argument in input dictionary!')
        return feed_dict

    def _build_inputs(self, input_batch):
        batch_size = self.config['training']['batch_size']

        if input_batch is None:
            self._inputs['encoder_inputs'] = tf.placeholder(
                shape=(batch_size, None), # batch_size, max_time
                dtype=tf.int32,
                name='encoder_inputs'
            )
            self._inputs['encoder_lengths'] = tf.placeholder(
                shape=(batch_size,),
                dtype=tf.int32,
                name='encoder_lengths'
            )
            self._inputs['decoder_inputs'] = tf.placeholder(
                shape=(batch_size, None), # batch_size, max_time
                dtype=tf.int32,
                name='decoder_inputs'
            )
            self._inputs['decoder_lengths'] = tf.placeholder(
                shape=(batch_size,),
                dtype=tf.int32,
                name='decoder_lengths'
            )

        else:
            encoder_inputs, encoder_lengths, decoder_inputs, decoder_lengths = input_batch
            encoder_inputs.set_shape([batch_size, None])
            decoder_inputs.set_shape([batch_size, None])
            encoder_lengths.set_shape([batch_size])
            decoder_lengths.set_shape([batch_size])

            self._inputs = {
                'encoder_inputs': encoder_inputs,
                'encoder_lengths': encoder_lengths,
                'decoder_inputs': decoder_inputs,
                'decoder_lengths': decoder_lengths
            }

        return self._inputs['encoder_inputs'], self._inputs['encoder_lengths'], \
               self._inputs['decoder_inputs'], self._inputs['decoder_lengths']

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
            state_size = self.config['encoder']['state_size']
            batch_size = self.config['training']['batch_size']

            # build cell
            if self.config['encoder']['cell_type'] == 'LSTM':
                encoder_cell = LSTMCell(state_size)
            elif self.config['encoder']['cell_type'] == 'GRU':
                encoder_cell = GRUCell(state_size)
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
                    encoder_state_c.set_shape([batch_size, state_size * 2])
                    encoder_state_h.set_shape([batch_size, state_size * 2])

                    encoder_final_state = LSTMStateTuple(encoder_state_c,
                                                         encoder_state_h)
                else:
                    encoder_final_state = tf.concat(
                        [fw_final_state, bw_final_state], 1)
                    encoder_final_state.set_shape([batch_size, state_size * 2])

            else:
                encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                    encoder_cell,
                    encoder_input_vectors,
                    sequence_length=encoder_lengths,
                    time_major=False,
                    dtype=tf.float32
                )
                encoder_final_state.set_shape([batch_size, state_size])

            return encoder_outputs, encoder_final_state

    def _build_decoder(self, encoder_outputs, encoder_final_state, encoder_lengths,
                             decoder_inputs, decoder_lengths):

        batch_size = self.config['training']['batch_size']
        beam_width = self.config['decoder']['beam_width']
        tiled_batch_size = batch_size * beam_width

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
            original_decoder_cell = decoder_cell

            with tf.variable_scope('beam_inputs'):
                tiled_encoder_outputs = tile_batch(encoder_outputs, beam_width)
                tiled_encoder_lengths = tile_batch(encoder_lengths, beam_width)

                if isinstance(encoder_final_state, LSTMStateTuple):
                    tiled_encoder_final_state_c = tile_batch(encoder_final_state.c, beam_width)
                    tiled_encoder_final_state_h = tile_batch(encoder_final_state.h, beam_width)
                    tiled_encoder_final_state = LSTMStateTuple(tiled_encoder_final_state_c,
                                                               tiled_encoder_final_state_h)
                else:
                    tiled_encoder_final_state = tile_batch(encoder_final_state, beam_width)

            if self.config['attention']['attention']:
                attention_name = self.config['attention']['attention_type']
                if attention_name == 'Bahdanau':
                    attention_fn = BahdanauAttention
                elif attention_name == 'Luong':
                    attention_fn = LuongAttention
                else:
                    raise ValueError

                with tf.variable_scope('attention'):
                    attention_mechanism = attention_fn(
                        self.config['attention']['attention_num_units'],
                        encoder_outputs,
                        encoder_lengths,
                        name="attention_fn"
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

                with tf.variable_scope('attention', reuse=True):
                    beam_attention_mechanism = attention_fn(
                        self.config['attention']['attention_num_units'],
                        tiled_encoder_outputs,
                        tiled_encoder_lengths,
                        name="attention_fn"
                    )
                    beam_decoder_cell = AttentionWrapper(
                        original_decoder_cell,
                        beam_attention_mechanism,
                        attention_layer_size=self.config['attention']['attention_depth'],
                        output_attention=True
                    )
                    tiled_decoder_initial_state = \
                        beam_decoder_cell.zero_state(tiled_batch_size, tf.float32).clone(
                            cell_state=tiled_encoder_final_state
                        )

            else:
                decoder_initial_state = encoder_final_state
                tiled_decoder_initial_state = decoder_cell.zero_state(tiled_batch_size, tf.float32)
                beam_decoder_cell = decoder_cell

        with tf.variable_scope('word_embedding', reuse=True):
            word_embedding = tf.get_variable(name="word_embedding")

        with tf.variable_scope('decoder'):
            out_func = layers_core.Dense(
                self.config['word']['num_word'], use_bias=False)

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
                output_layer=out_func,
            )

            decoder_outputs, decoder_state, decoder_sequence_lengths = \
                dynamic_decode(
                    decoder,
                    scope=tf.get_variable_scope()
                    # maximum_iterations=self.config['decoder']['max_iteration'] / TrainingHelper stops decode
                )

            tf.get_variable_scope().reuse_variables()

            start_tokens = tf.ones([batch_size], dtype=tf.int32) * self.EOS
            beam_decoder = BeamSearchDecoder(
                beam_decoder_cell,
                word_embedding,
                start_tokens,
                self.EOS,
                tiled_decoder_initial_state,
                beam_width,
                output_layer=out_func,
            )

            beam_decoder_outputs, beam_decoder_state, beam_decoder_sequence_lengths = \
                dynamic_decode(
                    beam_decoder,
                    scope=tf.get_variable_scope(),
                    maximum_iterations=self.config['decoder']['max_iteration']
                )

        decoder_results = {
            'decoder_outputs': decoder_outputs[0],
            'decoder_result_ids': decoder_outputs[1],
            'decoder_state': decoder_state,
            'decoder_sequence_lengths': decoder_sequence_lengths,
            'beam_decoder_result_ids': beam_decoder_outputs.predicted_ids,
            'beam_decoder_scores': beam_decoder_outputs.beam_search_decoder_output.scores,
            'beam_decoder_state': beam_decoder_state,
            'beam_decoder_sequence_outputs': beam_decoder_sequence_lengths
        }
        return decoder_results

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
                try:
                    grads_vars[i] = (tf.clip_by_norm(grad, 1.0), var)
                except ValueError:
                    print(var)
            apply_gradient_op = opt.apply_gradients(grads_vars)
            with tf.control_dependencies([apply_gradient_op]):
                train_op = tf.no_op(name='train_step')

        return train_op
