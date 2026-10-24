"""Fruit Edibility Detection System - Gradio web app.

Free and fully local:
- The open CLIP model names the fruit in the photo (zero-shot, works for many fruits).
- A classifier trained on this repo's dataset (train_freshness.py) judges fresh vs rotten.

Setup:  pip install -r requirements.txt
Run:    python app.py
Retrain the freshness classifier with python train_freshness.py (writes freshness_head.npz).
"""
import html
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image as PILImage
from transformers import CLIPModel, CLIPProcessor

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / 'Dataset' / 'dataset'
HEAD_PATH = BASE_DIR / 'freshness_head.npz'

# Rotten probability bands: in between, the app says it can't tell.
FRESH_BELOW, ROTTEN_ABOVE = 0.35, 0.65
NAME_SURE = 0.5  # below this name-match score, say we're not sure which fruit it is

FRUITS = {
    'apple': '🍎', 'green apple': '🍏', 'banana': '🍌', 'orange': '🍊', 'mandarin': '🍊', 'lemon': '🍋',
    'lime': '🍋', 'grapefruit': '🍊', 'pear': '🍐', 'peach': '🍑', 'nectarine': '🍑', 'apricot': '🍑',
    'plum': '🍑', 'cherry': '🍒', 'strawberry': '🍓', 'raspberry': '🍓', 'blackberry': '🫐', 'blueberry': '🫐',
    'grapes': '🍇', 'watermelon': '🍉', 'cantaloupe melon': '🍈', 'honeydew melon': '🍈', 'pineapple': '🍍',
    'mango': '🥭', 'papaya': '🥭', 'kiwi fruit': '🥝', 'coconut': '🥥', 'avocado': '🥑', 'tomato': '🍅',
    'pomegranate': '🍎', 'fig': '🍑', 'date fruit': '🌴', 'guava': '🍐', 'passion fruit': '🍑',
    'dragon fruit': '🐉', 'lychee': '🍒', 'longan': '🍒', 'rambutan': '🍒', 'durian': '🌵', 'jackfruit': '🍈',
    'persimmon': '🍅', 'star fruit': '⭐', 'quince': '🍐', 'cranberry': '🍒', 'gooseberry': '🍇',
    'mulberry': '🫐', 'olive': '🫒', 'custard apple': '🍏', 'tamarind': '🌰',
}
# 'plantain' is left out on purpose: CLIP called 30% of the test bananas plantains.
# Freshness was trained on these only; for other fruits it is untested.
TRAINED = {'apple', 'green apple', 'banana', 'orange'}
NOT_FRUIT = ['a vegetable', 'a person', 'an animal', 'a cooked meal', 'bread', 'a drink', 'an everyday object',
             'a room', 'a vehicle', 'a page of text']

# The larger CLIP names fruits better than the smaller one (84% on a 600-photo test sample); the
# freshness classifier was trained on the smaller CLIP's embeddings, so both are loaded.
NAME_CLIP = 'openai/clip-vit-large-patch14'

head = np.load(HEAD_PATH)
fresh_model = CLIPModel.from_pretrained(str(head['clip_name'])).eval()
fresh_processor = CLIPProcessor.from_pretrained(str(head['clip_name']))
name_model = CLIPModel.from_pretrained(NAME_CLIP).eval()
name_processor = CLIPProcessor.from_pretrained(NAME_CLIP)

labels = list(FRUITS) + NOT_FRUIT
prompts = [f'a photo of a {l}.' if l in FRUITS else f'a photo of {l}' for l in labels]
with torch.no_grad():
    text_emb = torch.nn.functional.normalize(
        name_model.get_text_features(**name_processor(text=prompts, return_tensors='pt', padding=True)).pooler_output,
        dim=-1).numpy()
logit_scale = name_model.logit_scale.exp().item()


def image_embedding(model, processor, img):
    feats = model.get_image_features(**processor(images=img, return_tensors='pt')).pooler_output
    return torch.nn.functional.normalize(feats, dim=-1).numpy()[0]


@torch.no_grad()
def analyze(image):
    img = PILImage.fromarray(image).convert('RGB')

    logits = logit_scale * text_emb @ image_embedding(name_model, name_processor, img)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    order = np.argsort(probs)[::-1]

    emb = image_embedding(fresh_model, fresh_processor, img)
    p_rotten = 1 / (1 + np.exp(-(head['coef'] @ emb + head['intercept'])))
    return [(labels[i], float(probs[i])) for i in order[:3]], float(p_rotten)


def card(kind, emoji, badge, title, body):
    return f"""
    <div class="result-card {kind}">
      <div class="verdict-row">
        <div class="big-emoji">{emoji}</div>
        <div>
          <div class="badge">{badge}</div>
          <h2>{html.escape(title)}</h2>
        </div>
      </div>
      {body}
      <p class="note">Based only on how the photo looks. This is not a food-safety check.</p>
    </div>
    """


EMPTY_HTML = """
<div class="result-card empty">
  <div class="big-emoji">🍽️</div>
  <h3>No fruit yet</h3>
  <p>Upload a photo of a fruit, snap one with your webcam, or try one of the examples.</p>
</div>
"""


def predict(image):
    if image is None:
        return EMPTY_HTML

    top, p_rotten = analyze(image)
    name, name_score = top[0]

    if name not in FRUITS:
        return card('unsure', '🔍', 'No fruit found', 'This doesn\'t look like a fruit',
                    f'<p class="signs">Closest match: {html.escape(name)}.</p>')

    title = name[0].upper() + name[1:]
    parts = []
    if name_score < NAME_SURE:
        others = ', '.join(html.escape(n) for n, _ in top[1:] if n in FRUITS)
        parts.append(f'<div class="warning">⚠️ Not sure which fruit this is'
                     f'{f". It could also be: {others}" if others else ""}.</div>')

    if p_rotten > ROTTEN_ABOVE:
        kind, badge, score = 'bad', 'Looks rotten', p_rotten
    elif p_rotten < FRESH_BELOW:
        kind, badge, score = 'good', 'Looks fresh', 1 - p_rotten
    else:
        kind, badge, score = 'unsure', 'Can\'t tell', max(p_rotten, 1 - p_rotten)
        parts.append('<div class="warning">⚠️ The model can\'t tell if this is fresh or rotten. '
                     'Try a clearer, closer photo in good light.</div>')

    parts.append(f"""
      <div class="conf-label"><span>Freshness model score</span><span>{'>99%' if score > 0.99 else f'{score:.0%}'}</span></div>
      <div class="conf-track"><div class="conf-fill" style="width:{score:.0%}"></div></div>""")

    if name not in TRAINED:
        parts.append(f'<p class="signs">ℹ️ The freshness model learned from apples, bananas and oranges only. '
                     f'It has not been tested on {html.escape(name)}, so treat this result with caution.</p>')

    return card(kind, FRUITS[name], badge, title, ''.join(parts))


CSS = """
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
#hero { text-align: center; padding: 28px 16px 8px; }
#hero h1 { font-size: 2.4rem; font-weight: 800; margin: 0;
  background: linear-gradient(90deg, #16a34a, #f59e0b, #ea580c);
  -webkit-background-clip: text; background-clip: text; color: transparent; }
#hero p { color: var(--body-text-color-subdued); font-size: 1.05rem; margin-top: 8px; }
#hero .privacy { font-size: .85rem; margin-top: 4px; }

.result-card { border-radius: 16px; padding: 22px; border: 1px solid var(--border-color-primary);
  background: var(--block-background-fill); }
.result-card.empty { text-align: center; padding: 48px 22px; color: var(--body-text-color-subdued); }
.result-card.empty h3 { margin: 8px 0 4px; color: var(--body-text-color); }
.result-card.good { border-top: 5px solid #16a34a; }
.result-card.bad { border-top: 5px solid #dc2626; }
.result-card.unsure { border-top: 5px solid #f59e0b; }
.verdict-row { display: flex; align-items: center; gap: 16px; margin-bottom: 16px; }
.big-emoji { font-size: 3.2rem; line-height: 1; }
.result-card h2 { margin: 6px 0 0; font-size: 1.6rem; }
.badge { display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: .85rem; }
.good .badge { background: #dcfce7; color: #166534; }
.bad .badge { background: #fee2e2; color: #991b1b; }
.unsure .badge { background: #fef3c7; color: #92400e; }
.warning { margin: 0 0 16px; padding: 10px 12px; border-radius: 10px; font-size: .9rem; font-weight: 600;
  background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
.conf-label { display: flex; justify-content: space-between; font-size: .9rem; font-weight: 600; margin-bottom: 6px; }
.conf-track { height: 10px; border-radius: 999px; background: var(--border-color-primary); overflow: hidden; }
.conf-fill { height: 100%; border-radius: 999px; }
.good .conf-fill { background: linear-gradient(90deg, #4ade80, #16a34a); }
.bad .conf-fill { background: linear-gradient(90deg, #f87171, #dc2626); }
.unsure .conf-fill { background: linear-gradient(90deg, #fcd34d, #f59e0b); }
.signs { margin: 16px 0 0; line-height: 1.5; }
.note { margin: 14px 0 0; font-size: .85rem; color: var(--body-text-color-subdued); }
#analyze-btn { font-size: 1.05rem; }
footer { display: none !important; }
"""

theme = gr.themes.Soft(
    primary_hue='green',
    secondary_hue='orange',
    radius_size='lg',
    font=[gr.themes.GoogleFont('Inter'), gr.themes.Font('system-ui'), gr.themes.Font('sans-serif')],
)

examples = [str(next((DATASET_PATH / 'test' / c).iterdir()))
            for c in ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']]

with gr.Blocks(theme=theme, css=CSS, title='Fruit Freshness Checker') as demo:
    gr.HTML("""
    <div id="hero">
      <h1>🍏 Fruit Freshness Checker</h1>
      <p>Names the fruit in your photo and checks whether it looks fresh or rotten.</p>
      <p class="privacy">Free and private: everything runs on this computer.</p>
    </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            image = gr.Image(type='numpy', label='Your fruit', sources=['upload', 'webcam', 'clipboard'], height=340)
            with gr.Row():
                clear_btn = gr.Button('Clear', variant='secondary')
                analyze_btn = gr.Button('🔍 Check freshness', variant='primary', elem_id='analyze-btn')
            gr.Examples(examples=examples, inputs=image, label='Try an example')

        with gr.Column(scale=1):
            result = gr.HTML(EMPTY_HTML)

    analyze_btn.click(predict, inputs=image, outputs=result)
    image.upload(predict, inputs=image, outputs=result)
    clear_btn.click(lambda: (None, EMPTY_HTML), outputs=[image, result])


if __name__ == '__main__':
    demo.launch(server_name='127.0.0.1', server_port=7860, inbrowser=True)
