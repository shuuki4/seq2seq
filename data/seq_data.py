from random import shuffle
import numpy as np


class SeqData:
    """
    Data class interface for seq2seq model
    """

    def __init__(self):
        """
        Data format should be [(from_sequence, to_sequence), ....]
        """
        self.max_length = None
        self.symbols = None

        self.train_sequences = list()
        self.val_sequences = list()

    @property
    def num_train_examples(self):
        return len(self.train_sequences)

    @property
    def num_val_examples(self):
        return len(self.val_sequences)

    @property
    def num_symbols(self):
        return len(self.symbols)

    @property
    def idx_to_symbol(self, symbol_idx):
        return self.symbols[symbol_idx]

    @property
    def initialized(self):
        return (self.max_length is not None) or (self.num_symbols is not None)

    def _next_batch(self, data, batch_idxs, batch_size):
        """
        Generate next batch. If len(batch_idxs) < batch_size, pad with empty sequences
        :param data: data list to process
        :param batch_idxs: idxs to process
        :return: next data dict of batch_size amount data
        """
        def _normalize_length(_data):
            return _data + [0] * (self.max_length - len(_data))

        def _empty_data():
            return _normalize_length([])

        assert(len(batch_idxs) <= batch_size)

        from_data, from_lengths, to_data, to_lengths = [], [], [], []
        for idx in batch_idxs:
            from_sequence, to_sequence = data[idx]
            from_lengths.append(len(from_sequence))
            to_lengths.append(len(to_sequence))
            from_data.append(_normalize_length(from_sequence))
            to_data.append(_normalize_length(to_sequence))

        for _ in range(batch_size - len(batch_idxs)):
            from_lengths.append(0)
            to_lengths.append(0)
            from_data.append(_empty_data())
            to_data.append(_empty_data())

        batch_data_dict = {
            'encoder_inputs': np.asarray(from_data, dtype=np.int32),
            'encoder_lengths': np.asarray(from_lengths, dtype=np.int32),
            'decoder_inputs': np.asarray(to_data, dtype=np.int32),
            'decoder_lengths': np.asarray(to_lengths, dtype=np.int32)
        }
        return batch_data_dict

    def _data_iterator(self, sequence, batch_size, random):
        idxs = list(range(len(sequence)))
        if random:
            shuffle(idxs)

        for start_idx in range(0, len(sequence), batch_size):
            end_idx = start_idx + batch_size
            yield self._next_batch(sequence, idxs[start_idx:end_idx], batch_size)

    def train_datas(self, batch_size=16, random=True):
        """
        Iterate through train data for single epoch
        :param batch_size: batch size
        :param random: if true, iterate randomly
        :return: train data iterator
        """
        assert self.initialized, "Dataset is not initialized!"
        return self._data_iterator(self.train_sequences, batch_size, random)

    def val_datas(self, batch_size=16, random=True):
        """
        Iterate through validaiton data for single epoch
        :param batch_size: batch size
        :param random: if true, iterate randomly
        :return: validation data iterator
        """
        assert self.initialized, "Dataset is not initialized!"
        return self._data_iterator(self.val_sequences, batch_size, random)

    def build(self):
        """
        Build data and save in self.train_sequences, self.val_sequences
        """
        raise NotImplementedError

    def load(self):
        """
        Load data and save in self.train_sequences, self.val_sequences
        """
        raise NotImplementedError
