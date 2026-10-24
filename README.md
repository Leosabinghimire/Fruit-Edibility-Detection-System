# Fruit Edibility Detection System

A machine learning system that looks at a photo of a fruit, names the fruit, and judges whether it looks fresh or rotten.
It is free and runs entirely on your computer; photos are never uploaded anywhere.

## How it works
- **Naming the fruit:** the open [CLIP](https://huggingface.co/openai/clip-vit-large-patch14) model (ViT-L/14) compares
  the photo with about 50 fruit names, plus a few non-fruit descriptions so it can say when there is no fruit.
- **Fresh or rotten:** a logistic-regression classifier trained on this repo's dataset, using embeddings from the smaller
  CLIP ViT-B/16 model (`train_freshness.py`, weights in `freshness_head.npz`).

## Usage
1. Clone the repository.
2. Install the required packages: `pip install -r requirements.txt`
3. Run the app: `python app.py`, then open http://127.0.0.1:7860
   The first run downloads the two CLIP models (about 2.3 GB).

To retrain the freshness classifier, run `python train_freshness.py`. It prints the test accuracy and writes `freshness_head.npz`.

## Measured accuracy
Measured on the dataset's test images, which contain only apples, bananas and oranges.

| Task | Accuracy |
|---|---|
| Fresh vs rotten (2,698 test images) | 97.3% (apples 95.8%, bananas 99.1%, oranges 97.1%) |
| Fresh vs rotten on a fruit left out of training (train on two fruits, test on the third) | 84–91% |
| Naming the fruit (random sample of 600 test images) | 86.2% (apples 79.0%, bananas 99.5%, oranges 80.0%) |

## Limitations
- Freshness was trained only on apples, bananas and oranges. For other fruits it has not been measured; the app shows a caution note.
- Naming accuracy for fruits outside the dataset has not been measured.
- Results are based only on how the photo looks. This is not a food-safety check.
- The training data includes some empty (0-byte) image files, which are skipped.

## Project files
- `app.py`: the Gradio web app.
- `train_freshness.py`: trains and evaluates the freshness classifier.
- `freshness_head.npz`: the trained freshness classifier.
- `project.ipynb`: the original notebook (a small TensorFlow CNN on 20×20-pixel images). It expects the dataset at
  `C:\Users\DELL\Downloads\dataset`; change `dataset_path` to `Dataset/dataset` to run it from the repo.
- `Dataset/`: training and test images (fresh and rotten apples, bananas and oranges).
