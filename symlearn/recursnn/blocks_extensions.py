from symlearn.blocks.bricks import Tanh, Linear, Logistic
from symlearn.blocks.bricks.base import lazy
from symlearn.blocks.bricks.recurrent import (SimpleRecurrent, recurrent, BaseRecurrent)
from symlearn.blocks.extensions import SimpleExtension
from symlearn.blocks.utils import unpack

from itertools import chain

import numpy as np
import logging
import copyreg

logger = logging.getLogger(__name__)

class AbstractStopCriterion(object):
    """
    base class for creating stopping criterion
    """
    def __init__(self, threshold):
        if threshold <= 0:
            raise ValueError
        self.current_best_cost = np.inf
        self.threshold = threshold
        self.stop_now = False
        self.abs_min = 0
        # TODO: maybe dealing with max by inverting the sign,
        # shifting if the min is not zero

    def _check_close_absmin(self):
        return(np.isclose(self.current_best_cost, self.abs_min))

    def _reset(self):
        self.current_best_cost = np.inf
        self.stop_now = False

    def is_stopping_now(self, current_cost, current_step):
        pass


class GeneralizedLossCriterion(AbstractStopCriterion):
    """
    implement generalized loss stopping criteiron where training will stop
    when the ratio of vadliation error of current iteration to minimal
    validation error (an estimate of optimal validation error) until current
    iteration given the threshold.
    """
    def __init__(self, threshold):
        super(__class__, self).__init__(threshold)

    def _reset(self):
        """
        reset current result
        """
        super(__class__, self)._reset()

    def is_stopping_now(self, current_cost, current_step):
        """
        core implementation of early stopping
        """
        if self.current_best_cost == np.inf or \
                current_cost <= self.current_best_cost:
            self.current_best_cost = current_cost
        if self._check_close_absmin() or \
                (current_cost / self.current_best_cost - 1) > \
                self.threshold * 100:
            self.stop_now = True
        return(self.stop_now)


class SuccessiveKStepCriterion(AbstractStopCriterion):
    """
    implement consecutive K steps (denoted as s in paper) stopping criteiron
    where training will stop when the current validation error is continue
    increasing for K steps in a row because the step the validation error starts
    climbing
    """
    def __init__(self, threshold):
        """
        @params threshold must be an integer represent as K consecutive steps
        before stopped
        """
        super(__class__, self).__init__(int(threshold))
        self.prev_cost = self.current_best_cost
        self.step_before_up = 0

    def _reset(self):
        """
        reset current result
        """
        super(__class__, self)._reset()
        self.prev_cost = self.current_best_cost
        self.step_before_up = 0

    def is_stopping_now(self, current_cost, current_step, epsilon=1e-5):
        """
        core implementation of early stopping
        """
        if self.current_best_cost == np.inf or \
                current_cost <= self.current_best_cost:
            self.current_best_cost = current_cost
        self.stop_now = self._check_close_absmin()
        # fix the step_before_up when the cost is increasing
        # update the step_before_up when the cost is not increasing
        if np.isclose(current_cost, self.prev_cost, atol=epsilon) or \
                current_cost > self.prev_cost:  # increasing
            if (current_step - self.step_before_up) > self.threshold:
                self.stop_now = True
        else:  # not increasing
            self.step_before_up = current_step
        self.prev_cost = current_cost
        return(self.stop_now)


class EarlyStopMonitoring(SimpleExtension):
    """
    pass the stop_now signal to the specified notification_name for
    FinishIfNotImprovement to stop
    """
    def __init__(self, record_name, notification_name=None,
                 strategy=None, **kwargs):

        kwargs.setdefault("before_training", True)
        kwargs.setdefault("after_epoch", True)
        super(__class__, self).__init__(**kwargs)
        self.record_name = record_name
        if not notification_name:
            notification_name = record_name + "_stopping_now"
        self.notification_name = notification_name
        self.strategy = strategy

    def _init_default_strategy(self):
        """
        find the relevant information for the strategy
        """
        terminator = None
        try:
            terminator = self.main_loop.find_extension(
                'FinishIfNoImprovementAfter')
        except ValueError as e:
            logger.info('find multiple candidates {}'.format(e))
            for ext in unpack([extension for extension in
                               self.main_loop.extensions
                               if extension.name ==
                               'FinishIfNoImprovementAfter'],
                               singleton=False):
                if ext.notification_name == self.notification_name:
                    terminator = ext

        if terminator is not None:
            if terminator.epochs is None:
                level = 'iterations'
            elif terminator.iterations is None:
                level = 'epochs'
            self.retrieve_key = level + '_done'
            self.step = getattr(terminator, level)
            setattr(terminator, level, 0)
        else:
            self.retrieve_key = 'epochs_done'
            self.step = 10

        if self.strategy is None:
            self.strategy = SuccessiveKStepCriterion(self.step)

    def before_training(self):
        """
        setting learning level and default training strategy
        """
        self._init_default_strategy()

    def dispatch(self, callback_name, *args):
        """
        modified from TrainingExtension for the purpose dealing with
        getting information about FinishIfNoImprovementAfter
        """
        if callback_name == 'before_training':
            getattr(self, str(callback_name))(*args)
        else:
            super(__class__, self).dispatch(callback_name, *args)

    def do(self, which_callback, *args):
        """
        core implementation of modifying notification
        """
        current_value = self.main_loop.log.current_row.get(self.record_name)
        current_step = self.main_loop.log.status[self.retrieve_key]
        if current_value is None:
            return
        if self.strategy.is_stopping_now(current_value, current_step):
            self.main_loop.log.current_row[self.notification_name] = True


class VanillaLSTM(BaseRecurrent):
    """
    from LSTM tutorial http://www.deeplearning.net/tutorial/lstm.html
    """
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, **kwargs):
        super(__class__, self).__init__()
        self.dim = dim

        if activation is None:
            activation = [Logistic(), Tanh()]

        for name in ['input', 'cell', 'forget', 'output']:
            if name == 'cell':
                act_func = activation[-1]
            else:
                act_func = activation[0]

            # constuct U
            self.children.append(SimpleRecurrent(dim=dim, name="%s_%s" % (name,
                        'state'), activation=act_func, **kwargs))

        self.children.append(activation[-1])

    def get_dim(self, name):
        if name in (VanillaLSTM.apply.sequences +
                    VanillaLSTM.apply.states):
            return self.dim
        return super(__class__, self).get_dim(name)

    def _allocate(self):
        """
        All allocation work are done when push allocation config. Only reassign
        parameters to class for convenience of access
        """
        self.parameters.extend(chain.from_iterable(child.parameters for child
                                in self.children))

    @recurrent(sequences=['inputs', 'mask'], states=['states', 'cells'],
               outputs=['states', 'cells'], contexts=[])
    def apply(self, inputs, states=None, cells=None, mask=None):
        state_results = {}

        for i in range(0, len(self.children) - 1):
            state_results[self.children[i].name] = \
                self.children[i].apply(
                    inputs=inputs.take(range(i * self.dim, (i + 1) * self.dim),
                        axis=inputs.ndim - 1), states=states, iterate=False)

        c_t = state_results['input_state'] * state_results['cell_state'] + \
            state_results['forget_state'] * cells
        h_t = state_results['output_state'] * self.children[-1].apply(c_t)

        return(h_t, c_t)
