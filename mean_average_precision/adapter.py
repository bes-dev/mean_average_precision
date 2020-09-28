"""
MIT License

Copyright (c) 2020 Sergei Belousov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np

class AdapterBase:
    """ Arguments Adapter for Metric.

    Arguments:
        metric_fn (MetricBase): metric function.
        value_config (dict): arguments of self..value(...) method.
    """
    def __init__(self, metric_fn, value_config=None):
        self.metric_fn = metric_fn
        self.value_config = value_config

    def add(self, preds, gt):
        """ Add sample to evaluation.

        Arguments:
            preds (np.array): predicted boxes.
            gt (np.array): ground truth boxes.
        """
        preds, gt = self._check_empty(preds, gt)
        preds = self._preds_adapter(preds)
        gt = self._gt_adapter(gt)
        return self.metric_fn.add(preds, gt)

    def value(self, *args, **kwargs):
        """ Evaluate Metric.

        Arguments:
            *args, **kwargs: metric specific arguments.

        Returns:
            metric (dict): evaluated metrics.
        """
        if self.value_config is not None:
            return self.metric_fn.value(**self.value_config)
        else:
            return self.metric_fn.value(*args, **kwargs)

    def reset(self):
        """ Reset stored data.
        """
        return self.metric_fn.reset()

    def _check_empty(self, preds, gt):
        """ Check empty arguments

        Arguments:
            preds (np.array): predicted boxes.
            gt (np.array): ground truth boxes.

        Returns:
            preds (np.array): predicted boxes.
            gt (np.array): ground truth boxes.
        """
        if not preds.size:
            preds = np.zeros((0, 6))
        if not gt.size:
            gt = np.zeros((0, 7))
        return preds, gt

    def _preds_adapter(self, preds):
        """ Preds adapter method.

        Should be implemented in child class.
        """
        raise NotImplemented

    def _gt_adapter(self, gt):
        """ Gt adapter method.

        Should be implemented in child class.
        """
        raise NotImplemented


class AdapterDefault(AdapterBase):
    """ Default implementation of adapter class.
    """
    def _preds_adapter(self, preds):
        return preds

    def _gt_adapter(self, gt):
        return gt
