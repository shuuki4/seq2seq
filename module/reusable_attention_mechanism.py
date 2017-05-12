import math

from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import BahdanauAttention, LuongAttention
from tensorflow.python.layers import core as layers_core


class ReusableBahdanauAttention(BahdanauAttention):

    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 normalize=False,
                 name="BahdanauAttention",
                 reuse=False):

        super(BahdanauAttention, self).__init__(
            query_layer=layers_core.Dense(
                num_units, name="query_layer", use_bias=False, _reuse=reuse),
            memory_layer=layers_core.Dense(
                num_units, name="memory_layer", use_bias=False, _reuse=reuse),
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            name=name)
        self._num_units = num_units
        self._normalize = normalize
        self._name = name


class ReusableLuongAttention(LuongAttention):

    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 scale=False,
                 name="LuongAttention",
                 reuse=False):

        # For LuongAttention, we only transform the memory layer; thus
        # num_units **must** match expected the query depth.
        super(LuongAttention, self).__init__(
            query_layer=None,
            memory_layer=layers_core.Dense(
                num_units, name="memory_layer", use_bias=False, _reuse=reuse),
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            name=name)
        self._num_units = num_units
        self._scale = scale
        self._name = name
