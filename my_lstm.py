import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell_impl
#linear = rnn_cell_impl._linear
linear = tf.contrib.layers.fully_connected
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import sigmoid, tanh
LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple


class MyLSTM(tf.nn.rnn_cell.BasicLSTMCell):

    def __call__(self, inputs, state, scope=None):
        """LSTM as mentioned in paper."""
        with vs.variable_scope(scope or "basic_lstm_cell"):
            # Parameters of gates are concatenated into one multiply for
            # efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = array_ops.split(
                    value=state, num_or_size_splits=2, axis=1)
            g = tf.concat([inputs, h],1)
            concat = linear([g], 4 * self._num_units, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = array_ops.split(
                value=concat,  num_or_size_splits=4, axis=1)

            new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
                     self._activation(j))
            new_h = self._activation(new_c) * sigmoid(o)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = array_ops.concat_v2([new_c, new_h], 1)
            return new_h, new_state
