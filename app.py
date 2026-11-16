"""Local NeuroGuard Gradio application; no training or test-set evaluation."""
import argparse
import os
from pathlib import Path

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ.setdefault('GRADIO_TEMP_DIR', str(Path(__file__).resolve().parent / 'tmp' / 'gradio'))

import gradio as gr
import torch

from src.inference import POINTS, Segmenter


CSS = """
.gradio-container {max-width: 1180px !important; margin: auto;}
#intro h1 {font-size: 2.8rem; letter-spacing: -.025em; margin-bottom: .2rem;}
#intro p {max-width: 72ch;}
#status {min-height: 70px;}
button.primary {background: #006b63 !important; color: white !important;}
button.primary:hover {background: #00564f !important;}
.tab-nav button.selected {color: #006b63 !important; border-color: #006b63 !important;}
button:focus-visible, input:focus-visible {outline: 3px solid #007c72; outline-offset: 3px;}
::selection {background: #b6e8de; color: #123b37;}
@media (max-width: 640px) {#intro h1 {font-size: 2rem;}}
"""


def build_app(segmenter):
    def run(path, choice, confirmed):
        try:
            flair, predictions = segmenter.predict(path, choice, confirmed)
            return (flair, [(p['overlay'], p['label']) for p in predictions],
                    [(p['mask'], p['label']) for p in predictions],
                    '**Complete.** Green marks predicted foreground; white marks foreground in the binary mask. '
                    'These are model predictions, not confirmed abnormalities. '
                    'No per-image accuracy or diagnostic confidence is available.')
        except ValueError as exc:
            return None, [], [], f'**Unable to segment.** {exc}'
        except Exception:
            return None, [], [], ('**Unable to segment.** Check the original checkpoint, TIFF format and '
                                   'installed dependencies. Restart the app if needed; see docs/demo.md.')

    with gr.Blocks(analytics_enabled=False, delete_cache=(60, 300), title='NeuroGuard') as demo:
        gr.Markdown('# NeuroGuard\nExplainable visual comparison of brain MRI abnormality segmentation approaches.', elem_id='intro')
        gr.Markdown('**This project is a prototype and is not medical advice.** '
                    'For radiologists and imaging researchers exploring a research comparison. '
                    'Use public, de-identified examples only.')
        with gr.Row():
            with gr.Column(scale=1, min_width=290):
                upload = gr.File(label='Compatible MRI slice · TIFF · up to 4 MiB',
                                 file_types=['.tif', '.tiff'], type='filepath')
                gr.Markdown('Use an **original LGG 256 × 256 × 3 uint8 TIFF**: stored pre-contrast / FLAIR / '
                            'post-contrast channels (including dataset substitutions). '
                            'JPEG, PNG, DICOM and arbitrary RGB images are unsupported. '
                            'File shape checks cannot establish MRI provenance.')
                confirmed = gr.Checkbox(label='I confirm this is a compatible public, de-identified LGG research image.', value=False)
                choice = gr.Dropdown(label='Model and frozen threshold', choices=list(POINTS), value=next(iter(POINTS)))
                submit = gr.Button('Run segmentation', variant='primary')
                gr.Markdown('**Recall has a cost.** On held-out empty-mask slices, FP rates were 27.02% '
                            '(BCE 0.5), 14.48% (BCE 0.8) and **85.79% (Tversky 0.5)**. '
                            'Tversky failed the validation FP constraint; 0.5 is its fallback. '
                            'These cohort results are not a confidence score for your image.')
            with gr.Column(scale=2, min_width=300):
                original = gr.Image(label='Uploaded MRI · stored FLAIR channel 1', type='numpy', image_mode='L', format='png', buttons=['download', 'fullscreen'], interactive=False)
                with gr.Tab('Segmentation overlay'):
                    overlays = gr.Gallery(label='Green = predicted region', format='png', columns=2, object_fit='contain', buttons=['download', 'fullscreen'], interactive=False)
                with gr.Tab('Binary mask'):
                    masks = gr.Gallery(label='White = predicted region · black = background', format='png', columns=2, object_fit='contain', buttons=['download', 'fullscreen'], interactive=False)
                status = gr.Markdown('Upload a compatible slice, confirm its provenance and run segmentation. '
                                     'No sample images or weights are bundled.', elem_id='status')
        with gr.Accordion('How to interpret the output', open=False):
            gr.Markdown('A U-Net produces one foreground probability per pixel; the chosen fixed threshold '
                        'creates the mask. There is no added cleanup. Missing highlights do not rule out disease; '
                        'highlights can be false positives. This 2D, single-dataset, single-seed study has '
                        'no external or clinical validation. No demographic fairness claim is supported. '
                        'The application cannot validate an image’s scanner, sequence or clinical suitability. '
                        'Uploads are processed by the local app; temporary files expire after five minutes '
                        '(checked each minute). Close the server and clear tmp/gradio after a session.')
        gr.Markdown('Dataset: [Mateusz Buda · LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) '
                    '· [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). '
                    'I directed the experimental design and analysis, using OpenAI ChatGPT and Codex '
                    'to accelerate implementation, debugging, testing and documentation.')
        submit.click(run, [upload, choice, confirmed], [original, overlays, masks, status],
                     concurrency_limit=1, api_name='segment')
        # Clear old results when inputs change, including after a failed run.
        for component in (upload, choice, confirmed):
            component.change(lambda: (None, [], [], 'Inputs changed. Run segmentation to see a new result.'),
                             outputs=[original, overlays, masks, status], queue=False)
    return demo.queue(max_size=4, default_concurrency_limit=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bce-checkpoint', default='local-assets/experiments/phase3-bce/bce-seed42-01/best.pt')
    parser.add_argument('--tversky-checkpoint', default='local-assets/experiments/phase4-tversky/tversky-seed42-01/best.pt')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    engine = Segmenter(args.bce_checkpoint, args.tversky_checkpoint, args.device)
    demo = build_app(engine)
    demo.launch(server_name='127.0.0.1', server_port=args.port, share=False,
                inbrowser=False, max_file_size='4mb', show_error=False,
                enable_monitoring=False, css=CSS, run_history=False,
                blocked_paths=[str(Path(__file__).resolve().parent / name) for name in
                               ('local-assets', 'local-context', '.git', '.venv', '.env', 'checkpoints')],
                theme=gr.themes.Base(primary_hue='teal', neutral_hue='slate',
                                     font=['Arial', 'sans-serif']))


if __name__ == '__main__':
    main()
