"""Smoke-test localhost Gradio with a reviewed image; never compute metrics."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
from gradio_client import Client, handle_file

from src.dataset import file_sha256
from src.inference import POINTS, Segmenter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', required=True, type=Path)
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    output = ROOT/'local-assets/app-smoke'
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    engine = Segmenter(ROOT/'local-assets/experiments/phase3-bce/bce-seed42-01/best.pt',
                       ROOT/'local-assets/experiments/phase4-tversky/tversky-seed42-01/best.pt')
    client = Client(f'http://127.0.0.1:{args.port}', download_files=str(output/'client'),
                    verbose=False, analytics_enabled=False)
    # Bound application check: one reviewed input, three frozen model/threshold
    # combinations. No manifest, annotations, selection or accuracy calculation.
    checked = []
    for choice in ('Compare · common thresholds', 'Compare · validation-selected / fallback'):
        flair, direct = engine.predict(args.scan, choice, True)
        result = client.predict(handle_file(str(args.scan.resolve())), choice, True, api_name='/segment')
        np.testing.assert_array_equal(np.asarray(Image.open(result[0])), flair)
        assert 'Complete' in result[3]
        for gallery, field in ((result[1], 'overlay'), (result[2], 'mask')):
            assert len(gallery) == len(direct)
            for item, expected in zip(gallery, direct, strict=True):
                np.testing.assert_array_equal(np.asarray(Image.open(item['image'])), expected[field])
                assert item['caption'] == expected['label']
        checked.append(choice)
    failed = client.predict(handle_file(str(args.scan.resolve())), next(iter(POINTS)), False, api_name='/segment')
    assert failed[0] is None and failed[1] == [] and failed[2] == [] and 'Confirm' in failed[3]
    report = {'kind':'application smoke test only; not experimental evaluation',
              'input_sha256':file_sha256(args.scan), 'device':'cpu',
              'checked':checked, 'http_output_equals_direct_inference':True,
              'unconfirmed_input_clears_outputs':True, 'performance_metrics_computed':False}
    report_path = output/f"smoke-{report['input_sha256'][:12]}.json"
    report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('PASS: real checkpoint inference via HTTP equals direct inference at all three frozen points.')
    print(f'Private report: local-assets/app-smoke/{report_path.name}. No accuracy metrics computed.')


if __name__ == '__main__':
    main()
