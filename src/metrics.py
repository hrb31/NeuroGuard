"""Explicit slice counts and two-stage patient-macro segmentation metrics."""
from collections import defaultdict
import numpy as np


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def slice_metrics(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or not prediction.size:
        raise ValueError("Prediction and target need matching nonempty shapes")
    if not set(np.unique(prediction)) <= {0, 1} or not set(np.unique(target)) <= {0, 1}:
        raise ValueError("Metrics require binary masks")
    p, y = prediction.astype(bool), target.astype(bool)
    tp, fp, fn, tn = (int(np.count_nonzero(a)) for a in
                      (p & y, p & ~y, ~p & y, ~p & ~y))
    positive, predicted = tp+fn, tp+fp
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "dice": ratio(2*tp, 2*tp+fp+fn) if positive or predicted else 1.0,
            "iou": ratio(tp, tp+fp+fn) if positive or predicted else 1.0,
            "recall": ratio(tp, positive),
            # Protocol convention: positive GT + empty prediction has precision=0.
            "precision": ratio(tp, predicted) if predicted else (0.0 if positive else None),
            "precision_denominator_zero": predicted == 0,
            "positive": positive > 0, "pixels": p.size}


def metric_row(prediction, target, patient, area_fraction, cutoffs):
    row = slice_metrics(prediction, target)
    row.update(patient=str(patient), size_group=("empty" if not row["positive"] else
               "small" if area_fraction <= cutoffs[0] else
               "medium" if area_fraction <= cutoffs[1] else "large"))
    return row


def patient_macro(rows):
    patients = defaultdict(list)
    for row in rows:
        patients[row["patient"]].append(row)
    result = {"patients": len(patients), "slices": len(rows)}
    for metric in ("dice", "recall", "precision", "iou"):
        means = [np.mean([r[metric] for r in group]) for group in patients.values()]
        result[metric] = float(np.mean(means)) if means else None
    return result


def aggregate(rows):
    positive = [r for r in rows if r["positive"]]
    empty = [r for r in rows if not r["positive"]]
    counts = {k: sum(r[k] for r in rows) for k in ("tp", "fp", "fn", "tn")}
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    return {"positive_patient_macro": patient_macro(positive),
            "size_groups": {g: patient_macro([r for r in positive if r["size_group"] == g])
                            for g in ("small", "medium", "large")},
            "empty": {"slices": len(empty), "patients": len({r['patient'] for r in empty}),
                      "false_positive_rate": ratio(sum(r["fp"] > 0 for r in empty), len(empty)),
                      "mean_foreground_fraction": float(np.mean([r['fp']/r['pixels'] for r in empty])) if empty else None,
                      "pooled_background_fp_rate": ratio(sum(r['fp'] for r in empty), sum(r['pixels'] for r in empty))},
            "pooled_all_slices": dict(counts, dice=ratio(2*tp, 2*tp+fp+fn),
                recall=ratio(tp, tp+fn), precision=ratio(tp, tp+fp), iou=ratio(tp, tp+fp+fn),
                pixel_accuracy=ratio(tp+tn, tp+tn+fp+fn)),
            "precision_denominator_zero_slices": sum(r['precision_denominator_zero'] for r in rows),
            "slices": len(rows)}


def select_threshold(reports, fp_limit=.1):
    eligible = [(float(t), r) for t, r in reports.items()
                if r['positive_patient_macro']['dice'] is not None]
    if not eligible:
        return {"threshold": .5, "status": "no_positive_patients"}
    has_empty = eligible[0][1]['empty']['slices'] > 0
    feasible = [(t, r) for t, r in eligible if not has_empty or
                r['empty']['false_positive_rate'] <= fp_limit]
    if not feasible:
        return {"threshold": .5, "status": "fp_constraint_unmet_fallback"}
    threshold, _ = max(feasible, key=lambda pair: (
        pair[1]['positive_patient_macro']['dice'],
        -pair[1]['empty']['false_positive_rate'] if has_empty else 0, pair[0]))
    return {"threshold": threshold, "status": "constraint_met" if has_empty else "fp_constraint_unevaluable"}
