import random

from data.seq_data import SeqData
from data.corpus.vectorizer import Vectorizer
from util import log


class CorpusData(SeqData):
    """
    Data class for corpus
    """

    def __init__(self, max_length=10):
        super(CorpusData, self).__init__()
        self.max_length = max_length
        self.vectorizer = Vectorizer()

    def load(self, corpus_path=None, vocab_path=None, min_length=3):

        assert corpus_path is not None and vocab_path is not None

        self.vectorizer.load(vocab_path)

        log.infov('Start loading data...')
        potential_data = []

        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                send, recv = line.strip().split('\t')
                send = self.vectorizer.encode(send)
                recv = self.vectorizer.encode(recv)
                if len(send) > self.max_length or len(recv) > self.max_length \
                        or len(send) < min_length or len(recv) < min_length:
                    continue
                if self.vectorizer.UNK in send:
                    continue

                potential_data.append((send, recv))

        log.infov('Loaded {} corpus data!'.format(len(potential_data)))

        random.shuffle(potential_data)
        train_val_cut = int(0.9 * len(potential_data))
        self.train_sequences = potential_data[:train_val_cut]
        self.val_sequences = potential_data[train_val_cut:]
        self.symbols = self.vectorizer.idx2vocab
