import collections
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import variable_scope as vs

_checked_scope = core_rnn_cell_impl._checked_scope
_linear = core_rnn_cell_impl._linear

class RANCell(RNNCell):
  """Recurrent Additive Networks (cf. https://arxiv.org/abs/1705.07393)."""

  def __init__(self, num_units, input_size=None, activation=tanh, reuse=None):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation
    self._reuse = reuse

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """
    Deriving RAN from GRU [Sec 4.4]
    """
    with _checked_scope(self, scope or "ran_cell", reuse=self._reuse):
      with vs.variable_scope("input_x"):
        lx = _linear(inputs, self._num_units, True)

      with vs.variable_scope("input_h"):
        lh = _linear(state, self._num_units, True)

      i = tf.nn.sigmoid(lh + lx)

      with vs.variable_scope("output_x"):
        lx = _linear(inputs, self._num_units, True)

      with vs.variable_scope("output_h"):
        lh = _linear(state, self._num_units, True)

      o = tf.nn.sigmoid(lh + lx)

      with vs.variable_scope("candidate"):
        c = self._activation(_linear([state, inputs], self._num_units, True))
        c = i * c + (1 - i) * state

      new_h = c * o

    return new_h, new_h
