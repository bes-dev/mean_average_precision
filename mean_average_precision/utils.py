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

def sort_by_col(array, idx=1):
    """Sort np.array by column."""
    order = np.argsort(array[:, idx])[::-1]
    return array[order]

def compute_precision_recall(tp, fp, n_positives):
    """ Compute Preision/Recall.

    Arguments:
        tp (np.array): true positives array.
        fp (np.array): false positives.
        n_positives (int): num positives.

    Returns:
        precision (np.array)
        recall (np.array)
    """
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / max(float(n_positives), 1)
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return precision, recall

def compute_average_precision(precision, recall):
    """ Compute Avearage Precision by all points.

    Arguments:
        precision (np.array): precision values.
        recall (np.array): recall values.

    Returns:
        average_precision (np.array)
    """
    precision = np.concatenate(([0.], precision, [0.]))
    recall = np.concatenate(([0.], recall, [1.]))
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])
    ids = np.where(recall[1:] != recall[:-1])[0]
    average_precision = np.sum((recall[ids + 1] - recall[ids]) * precision[ids + 1])
    return average_precision

def compute_average_precision_with_recall_thresholds(precision, recall, recall_thresholds):
    """ Compute Avearage Precision by specific points.

    Arguments:
        precision (np.array): precision values.
        recall (np.array): recall values.
        recall_thresholds (np.array): specific recall thresholds.

    Returns:
        average_precision (np.array)
    """
    average_precision = 0.
    for t in recall_thresholds:
        p = np.max(precision[recall >= t]) if np.sum(recall >= t) != 0 else 0
        average_precision = average_precision + p / recall_thresholds.size
    return average_precision

def compute_iou(pred, gt):
    """ Calculates IoU (Jaccard index) of two sets of bboxes:
            IOU = pred ∩ gt / (area(pred) + area(gt) - pred ∩ gt)

        Parameters:
            Coordinates of bboxes are supposed to be in the following form: [x1, y1, x2, y2]
            pred (np.array): predicted bboxes
            gt (np.array): ground truth bboxes

        Return value:
            iou (np.array): intersection over union
    """
    def get_box_area(box):
        return (box[:, 2] - box[:, 0] + 1.) * (box[:, 3] - box[:, 1] + 1.)

    _gt = np.tile(gt, (pred.shape[0], 1))
    _pred = np.repeat(pred, gt.shape[0], axis=0)

    ixmin = np.maximum(_gt[:, 0], _pred[:, 0])
    iymin = np.maximum(_gt[:, 1], _pred[:, 1])
    ixmax = np.minimum(_gt[:, 2], _pred[:, 2])
    iymax = np.minimum(_gt[:, 3], _pred[:, 3])

    width = np.maximum(ixmax - ixmin + 1., 0)
    height = np.maximum(iymax - iymin + 1., 0)

    intersection_area = width * height
    union_area = get_box_area(_gt) + get_box_area(_pred) - intersection_area
    iou = (intersection_area / union_area).reshape(pred.shape[0], gt.shape[0])
    return iou

def compute_match_table(preds, gt, img_id):
    """ Compute match table.

    Arguments:
        preds (np.array): predicted boxes.
        gt (np.array): ground truth boxes.
        img_id (int): image id

    Returns:
        match_table (np.array)


    Input format:
        preds: [xmin, ymin, xmax, ymax, class_id, confidence]
        gt: [xmin, ymin, xmax, ymax, class_id, difficult]

    Output format:
        match_table: [img_id, confidence, match_id, iou, difficult]
    """
    iou = compute_iou(preds, gt)
    if preds.shape[0] > 0:
        match_table = np.zeros((preds.shape[0], 5))
        match_table[:, 0] = img_id
        match_table[:, 1] = preds[:, 5]
        if iou.shape[0] > 0 and iou.shape[1] > 0:
            match_table[:, 2] = iou.argmax(axis=1)
            match_table[:, 3] = iou.max(axis=1)
            match_table[:, 4] = gt[match_table[:, 2].astype(int), 5]
        else:
            match_table[:, 2] = -1
            match_table[:, 3] = -1
    else:
        match_table = np.zeros((0, 5))
    return match_table
