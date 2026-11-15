import copy
import json
from pathlib import Path

import pytest
pytest.importorskip('torch')
from src.reporting import compare_validation


def test_comparison_units_missing_values_and_failure():
    reference = json.loads(Path('docs/bce-validation-reference.json').read_text())
    def report(dice, fp):
        return {'slices': 621, 'positive_patient_macro': {'slices': 209, 'dice': dice, 'recall': .8, 'precision': .7},
                'empty': {'false_positive_rate': fp},
                'size_groups': {'small': {'dice': None, 'recall': .6}}}
    reports = {'default_threshold': .5, 'default': report(.74794, .2),
               'selected': report(.69258, .09),
               'selection': {'threshold': .85, 'status': 'constraint_met'}}
    comparison = compare_validation(reports, reference)
    default = comparison['operating_points']['default']
    assert default['difference_pp']['dice'] == pytest.approx(1.)
    assert default['difference_pp']['small_dice'] is None
    assert default['research_fp_guardrail_met'] is False
    selected = comparison['operating_points']['selected']
    assert selected['tversky_threshold'] == .85
    assert selected['research_fp_guardrail_met'] is True
    changed = copy.deepcopy(reports)
    changed['default']['slices'] = 1
    with pytest.raises(ValueError, match='cohort'):
        compare_validation(changed, reference)
