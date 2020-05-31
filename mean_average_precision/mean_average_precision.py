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
from .utils import *

class MeanAveragePrecision:
    """ Mean Average Precision for object detection.

    Arguments:
        num_classes (int): number of classes.
        iou_thresholds (list of float): IOU thresholds.
        recall_thresholds (np.array or None): specific recall thresholds to the
                                              computation of average precision.
    """
    def __init__(self, num_classes, iou_thresholds=[0.5], recall_thresholds=None):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds
        if isinstance(self.iou_thresholds, float):
            self.iou_thresholds = [self.iou_thresholds]
        self.recall_thresholds = recall_thresholds
        self._init()

    def reset(self):
        """Reset stored data."""
        self._init()

    def add(self, preds, gt):
        """ Add sample to evaluation.

        Arguments:
            preds (np.array): predicted boxes.
            gt (np.array): ground truth boxes.

        Input format:
            preds: [xmin, ymin, xmax, ymax, class_id, confidence]
            gt: [xmin, ymin, xmax, ymax, class_id, difficult]
        """
        class_counter = np.zeros((1, self.num_classes), dtype=np.int32)
        for c in range(self.num_classes):
            preds_c = preds[preds[:, 4] == c]
            gt_c = gt[gt[:, 4] == c]
            class_counter[0, c] = gt_c.shape[0]
            match_table = compute_match_table(preds_c, gt_c, self.imgs_counter)
            self.match_table[c] = np.concatenate((self.match_table[c], match_table), axis=0)
        self.imgs_counter = self.imgs_counter + 1
        self.class_counter = np.concatenate((self.class_counter, class_counter), axis=0)

    def value(self):
        """ Evaluate Mean Average Precision.

        Returns:
            metric (dict): evaluated metrics.

        Output format:
            {
                "mAP": float.
                "<iou_threshold_0>":
                {
                    "<cls_id>":
                    {
                        "ap": float,
                        "precision": np.array,
                        "recall": np.array,
                    }
                },
                ...
                "<iou_threshold_N>":
                {
                    "<cls_id>":
                    {
                        "ap": float,
                        "precision": np.array,
                        "recall": np.array,
                    }
                }
            }
        """
        metric = {}
        aps = np.zeros((0, self.num_classes), dtype=np.float32)
        for t in self.iou_thresholds:
            metric[t] = {}
            aps_t = np.zeros((1, self.num_classes), dtype=np.float32)
            for class_id in range(self.num_classes):
                aps_t[0, class_id], precision, recall = self._evaluate_class(class_id, t)
                metric[t][class_id] = {}
                metric[t][class_id]["ap"] = aps_t[0, class_id]
                metric[t][class_id]["precision"] = precision
                metric[t][class_id]["recall"] = recall
            aps = np.concatenate((aps, aps_t), axis=0)
        metric["mAP"] = aps.mean(axis=1).mean(axis=0)
        return metric

    def _evaluate_class(self, class_id, iou_threshold):
        """ Evaluate class.

        Arguments:
            class_id (int): index of evaluated class.
            iou_threshold (float): iou threshold.

        Returns:
            average_precision (np.array)
            precision (np.array)
            recall (np.array)
        """
        table = sort_by_col(self.match_table[class_id], idx=1)
        matched_ind = {}
        nd = table.shape[0]
        tp = np.zeros(nd, dtype=np.float64)
        fp = np.zeros(nd, dtype=np.float64)
        for d in range(nd):
            img_id, conf, gt_id, iou, difficult = table[d]
            img_id = int(img_id)
            gt_id = int(gt_id)
            difficult = int(difficult)
            if iou > iou_threshold:
                if not difficult:
                    if not img_id in matched_ind:
                        matched_ind[img_id] = []
                    if not gt_id in matched_ind[img_id]:
                        tp[d] = 1
                        matched_ind[img_id].append(gt_id)
                    else:
                        fp[d] = 1
            else:
                fp[d] = 1
        precision, recall = compute_precision_recall(tp, fp, self.class_counter[:, class_id].sum())
        if self.recall_thresholds is None:
            average_precision = compute_average_precision(precision, recall)
        else:
            average_precision = compute_average_precision_with_recall_thresholds(precision, recall,
                                                                                 self.recall_thresholds)
        return average_precision, precision, recall

    def _init(self):
        """ Initialize internal state."""
        self.imgs_counter = 0
        self.class_counter = np.zeros((0, self.num_classes), dtype=np.int32)
        self.match_table = []
        for i in range(self.num_classes):
            self.match_table.append(np.zeros((0, 5), dtype=np.float32))
