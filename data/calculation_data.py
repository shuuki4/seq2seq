from data.seq_data import SeqData
from random import randint


class CalculationSeqData(SeqData):
    """
    Sequence data for add/subtract calculation task
    """

    @staticmethod
    def _make_data_instance(int_max, symbol_mapper):
        n1 = randint(1, int_max)
        n2 = randint(1, int_max)
        operator_rand = randint(0, 1)
        if operator_rand == 0:
            n3 = n1 + n2
            operator_char = '+'
        else:
            n3 = n1 - n2
            operator_char = '-'

        from_sequence = [symbol_mapper[c] for c in str(n1) + operator_char + str(n2)]
        to_sequence = [symbol_mapper[c] for c in str(n3)]
        return from_sequence, to_sequence

    def build(self, max_digit=4, num_train=100000, num_val=10000):
        """
        build simple summation task sequence data
        :param max_digit: maximum number of digits
        :param num_train: number of training examples
        :param num_val: number of validation examples
        """
        assert max_digit >= 1

        self.symbols = "_0123456789+-"  # 0 for pad
        symbol_mapper = {c: i for i, c in enumerate(self.symbols)}

        int_max = 1
        for _ in range(max_digit):
            int_max *= 10
        int_max -= 1

        self.train_sequences = list()
        self.val_sequences = list()
        for _ in range(num_train):
            self.train_sequences.append(self._make_data_instance(int_max, symbol_mapper))
        for _ in range(num_val):
            self.val_sequences.append(self._make_data_instance(int_max, symbol_mapper))

        self.max_length = 2 * max_digit + 1
