from .mean_average_precision_2d import MeanAveragePrecision2d
from .multiprocessing import MetricMultiprocessing

metrics_dict = {
    'map_2d': MeanAveragePrecision2d
}

class MetricBuilder:
    @staticmethod
    def get_metrics_list():
        """ Get evaluation metrics list."""
        return list(metrics_dict.keys())

    @staticmethod
    def build_evaluation_metric(metric_type, async_mode=False, *args, **kwargs):
        """ Build evaluation metric.

        Arguments:
            metric_type (str): type of evaluation metric.
            async_mode (bool): use multiprocessing metric.

        Returns:
            metric_fn (MetricBase): instance of the evaluation metric.
        """
        assert metric_type in metrics_dict, "Unknown metric_type"
        if not async_mode:
            metric_fn = metrics_dict[metric_type](*args, **kwargs)
        else:
            metric_fn = MetricMultiprocessing(metrics_dict[metric_type], *args, **kwargs)
        return metric_fn
