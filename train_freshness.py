"""Train the fresh/rotten classifier used by app.py and report its accuracy.

Uses the free CLIP model (runs locally) to turn each photo into an embedding, then fits a
logistic-regression classifier on the dataset's fresh/rotten labels.

Run:  python train_freshness.py
Writes freshness_head.npz (the classifier weights) next to this file.
"""
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from transformers import CLIPModel, CLIPProcessor

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / 'Dataset' / 'dataset'
HEAD_PATH = BASE_DIR / 'freshness_head.npz'
CACHE_PATH = BASE_DIR / 'clip_embeddings_cache.npz'
CLIP_NAME = 'openai/clip-vit-base-patch16'
TRAIN_PER_CLASS = 500  # a random sample is enough for a linear classifier and keeps CPU time down
CLASSES = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']


def list_images(split, per_class=None):
    paths, rotten, fruit = [], [], []
    for c in CLASSES:
        folder = DATASET_PATH / split / c
        files = [folder / f for f in sorted(os.listdir(folder))
                 if (folder / f).stat().st_size > 0]  # the dataset contains some empty files
        if per_class and len(files) > per_class:
            files = random.Random(0).sample(files, per_class)
        for p in files:
            paths.append(p)
            rotten.append(int(c.startswith('rotten')))
            fruit.append(c.removeprefix('fresh').removeprefix('rotten'))
    return paths, np.array(rotten), np.array(fruit)


@torch.no_grad()
def embed(model, processor, paths, batch_size=64):
    out = []
    for i in range(0, len(paths), batch_size):
        images = [Image.open(p).convert('RGB') for p in paths[i:i + batch_size]]
        feats = model.get_image_features(**processor(images=images, return_tensors='pt')).pooler_output
        out.append(torch.nn.functional.normalize(feats, dim=-1).numpy())
        print(f'  {min(i + batch_size, len(paths))}/{len(paths)}', flush=True)
    return np.concatenate(out)


def main():
    train_paths, y_train, fruit_train = list_images('train', per_class=TRAIN_PER_CLASS)
    test_paths, y_test, fruit_test = list_images('test')

    if CACHE_PATH.exists():
        cache = np.load(CACHE_PATH)
        X_train, X_test = cache['X_train'], cache['X_test']
    else:
        model = CLIPModel.from_pretrained(CLIP_NAME).eval()
        processor = CLIPProcessor.from_pretrained(CLIP_NAME)
        print(f'Embedding {len(train_paths)} training images')
        X_train = embed(model, processor, train_paths)
        print(f'Embedding {len(test_paths)} test images')
        X_test = embed(model, processor, test_paths)
        np.savez(CACHE_PATH, X_train=X_train, X_test=X_test)

    clf = LogisticRegression(max_iter=5000).fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(f'\nTest accuracy (fresh vs rotten): {np.mean(pred == y_test):.1%} on {len(y_test)} images')
    for f in ['apples', 'banana', 'oranges']:
        m = fruit_test == f
        print(f'  {f:8s} {np.mean(pred[m] == y_test[m]):.1%} ({m.sum()} images)')

    print('\nUnseen-fruit check (train on two fruits, test on the third):')
    for f in ['apples', 'banana', 'oranges']:
        held_out = LogisticRegression(max_iter=5000).fit(X_train[fruit_train != f], y_train[fruit_train != f])
        m = fruit_test == f
        print(f'  {f:8s} {np.mean(held_out.predict(X_test[m]) == y_test[m]):.1%}')

    np.savez(HEAD_PATH, coef=clf.coef_[0], intercept=clf.intercept_[0], clip_name=CLIP_NAME)
    print(f'\nSaved {HEAD_PATH.name}')


if __name__ == '__main__':
    main()
