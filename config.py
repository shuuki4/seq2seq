
class Config:

    def __init__(self,
                 is_training, num_words,
                 learning_rate=0.01, minimum_learning_rate=1e-5,
                 batch_size=128, decay_steps=1e4, decay_factor=0.3,
                 word_embedding_dim=100, cell_type='LSTM',
                 rnn_state_size=100, encoder_bidirection=True,
                 beam_width=5, max_iteration=10,
                 attention=True, attention_type='Bahdanau',
                 attention_num_units=100, attention_depth=100):

        self.config = dict()
        assert cell_type in ['LSTM', 'GRU']

        # training config
        self.config['training'] = training_config = dict()
        training_config['is_training'] = is_training
        training_config['learning_rate'] = learning_rate
        training_config['minimum_learning_rate'] = minimum_learning_rate
        training_config['batch_size'] = batch_size
        training_config['decay_steps'] = decay_steps
        training_config['decay_factor'] = decay_factor

        # word embedding config
        self.config['word'] = word_config = dict()
        word_config['num_word'] = num_words
        word_config['embedding_dim'] = word_embedding_dim

        # encoder config
        self.config['encoder'] = encoder_config = dict()
        encoder_config['cell_type'] = cell_type
        encoder_config['state_size'] = rnn_state_size
        encoder_config['bidirection'] = encoder_bidirection

        # decoder config
        self.config['decoder'] = decoder_config = dict()
        decoder_config['cell_type'] = cell_type
        decoder_config['state_size'] = rnn_state_size
        decoder_config['beam_width'] = beam_width
        decoder_config['max_iteration'] = max_iteration

        # attention config
        assert attention_type in ['Bahdanau', 'Luong']
        self.config['attention'] = attention_config = dict()
        attention_config['attention'] = attention
        attention_config['attention_type'] = attention_type
        attention_config['attention_num_units'] = attention_num_units
        attention_config['attention_depth'] = attention_depth

    def __getitem__(self, keys):
        if type(keys) == str:
            try:
                return self.config[keys]
            except KeyError as e:
                raise KeyError('Wrong key {} for config'.format(keys))

        elif type(keys) == list or type(keys) == tuple:
            assert len(keys) == 2
            try:
                return self.config[keys[0]][keys[1]]
            except KeyError as e:
                raise KeyError('Wrong key {} for config'.format(keys))
