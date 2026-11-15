import numpy as np
import pytest
from src.metrics import aggregate, metric_row, select_threshold, slice_metrics


def row(patient, prediction, target, area=.1):
    return metric_row(np.array(prediction), np.array(target), patient, area, [.1, .2])


def test_empty_denominators_and_false_positives():
    both = slice_metrics([0, 0], [0, 0])
    assert both['dice'] == both['iou'] == 1
    assert both['precision'] is both['recall'] is None
    missed = slice_metrics([0, 0], [1, 0])
    assert [missed[k] for k in ('dice', 'recall', 'precision', 'iou')] == [0]*4
    fp = slice_metrics([1, 0], [0, 0])
    assert fp['precision'] == fp['dice'] == 0 and fp['recall'] is None
    with pytest.raises(ValueError):
        slice_metrics([.5], [1])


def test_two_stage_aggregation_and_size_boundaries():
    rows = [row('A', [1], [1]), row('B', [0], [1]), row('B', [0], [1]),
            row('B', [1], [0], 0), row('C', [0], [0], 0)]
    report = aggregate(rows)
    primary = report['positive_patient_macro']
    assert primary['dice'] == primary['recall'] == primary['precision'] == .5
    assert primary['patients'] == 2 and primary['slices'] == 3
    assert report['empty']['false_positive_rate'] == .5
    assert report['empty']['mean_foreground_fraction'] == .5
    assert report['size_groups']['small']['dice'] == .5
    assert report['size_groups']['medium']['dice'] is None
    assert row('A', [1], [1], .2)['size_group'] == 'medium'
    assert row('A', [1], [1], .21)['size_group'] == 'large'
    assert aggregate([row('A', [0], [0], 0)])['positive_patient_macro']['dice'] is None
    assert aggregate([row('A', [1], [1])])['empty']['false_positive_rate'] is None
    assert aggregate([row('A', [0], [0], 0)])['pooled_all_slices']['dice'] is None


def test_trivial_baseline_accuracy_can_mislead():
    report = aggregate([row('A', [0]*100, [1]+[0]*99)])
    assert report['positive_patient_macro']['dice'] == 0
    assert report['pooled_all_slices']['pixel_accuracy'] == .99
    assert report['pooled_all_slices']['precision'] is None


def threshold_report(dice, fp, empty=10):
    return {'positive_patient_macro': {'dice': dice}, 'empty': {'false_positive_rate': fp, 'slices': empty}}


def test_threshold_selection_constraints_ties_and_fallback():
    reports = {.5: threshold_report(.8, .2), .6: threshold_report(.7, .1),
               .7: threshold_report(.7, .05), .8: threshold_report(.7, .05)}
    assert select_threshold(reports)['threshold'] == .8
    assert select_threshold({.5: reports[.5]})['status'] == 'fp_constraint_unmet_fallback'
    assert select_threshold({.5: threshold_report(.5, None, 0)})['status'] == 'fp_constraint_unevaluable'
    assert select_threshold({.5: threshold_report(None, 0)})['status'] == 'no_positive_patients'
