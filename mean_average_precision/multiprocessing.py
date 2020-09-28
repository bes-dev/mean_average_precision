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

from .metric_base import MetricBase
from multiprocessing import Process, Queue
from multiprocessing.managers import BaseManager

def create_metric_fn(metric_type, *args, **kwargs):
    """ Create multiprocessing version of metric function.

    Arguments:
        metric_type (dtype): type of metric function.
        *args, **kwargs: metric specific arguments.

    Returns:
        metric_fn: instance of metric.
    """
    if not hasattr(BaseManager, str(metric_type)):
        BaseManager.register(str(metric_type), metric_type)
    manager = BaseManager()
    manager.start()
    return getattr(manager, str(metric_type))(*args, **kwargs)


class MetricMultiprocessing(MetricBase):
    """ Implements parallelism at the metric level.

    This container provides an asynchronous interface for the
    metric class using multiprocessing. It provides functionality
    to async computation 'per frame' part of the metric in parallel
    with next frame inference.

    Arguments:
        metric_type (dtype): type of metric function.
        *args, **kwargs: metric specific arguments.
    """
    def __init__(self, metric_type, *args, **kwargs):
        super().__init__()
        self.metric_fn = create_metric_fn(metric_type, *args, **kwargs)
        self.proc = None
        self.queue = None
        self.is_start = False

    def add(self, preds, gt):
        """ Add sample to evaluation.
        Asynchronous wrapper for 'add' method.

        Arguments:
            preds (np.array): predicted boxes.
            gt (np.array): ground truth boxes.

        Input format:
            preds: [xmin, ymin, xmax, ymax, class_id, confidence]
            gt: [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
        """
        if not self.is_start:
            self.start()
        self.queue.put((preds, gt))

    def value(self, *args, **kwargs):
        """ Evaluate Metric.
        Asynchronous wrapper for 'value' method.

        Arguments:
            *args, **kwargs: metric specific arguments.

        Returns:
            metric (dict): evaluated metrics.
        """
        if self.is_start:
            self.stop()
        return self.metric_fn.value(*args, **kwargs)

    def reset(self):
        """ Reset stored data.
        Asynchronous wrapper for 'reset' method.
        """
        self._reset_proc()
        self.metric_fn.reset()

    def start(self):
        """ Start child process."""
        self._init_proc()

    def stop(self):
        """ Stop child process."""
        self._reset_proc()

    @staticmethod
    def _proc_loop(metric_fn, queue):
        """ Body of multiprocessing add."""
        while True:
            preds, gt = queue.get()
            if preds is None and gt is None:
                break
            metric_fn.add(preds, gt)

    def _init_proc(self):
        """ Initialize child process."""
        if self.queue is None:
            self.queue = Queue()
        if self.proc is None:
            self.proc = Process(
                target=self._proc_loop,
                args=[self.metric_fn, self.queue],
                daemon=True
            )
            self.proc.start()
        self.is_start = True

    def _reset_proc(self):
        """ Reset child process."""
        if self.proc is not None:
            self.add(None, None)
            self.proc.join()
            self.proc = None
        if self.queue is not None:
            self.queue = None
        self.is_start = False
