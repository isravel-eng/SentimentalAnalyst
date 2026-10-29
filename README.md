# Sentiment Analyst

A beginner-friendly sentiment classifier with a Google Colab training workflow and a small Streamlit app. Train from Sentiment140 or Keras IMDB in Colab, save the model and evaluation image into this GitHub repository, then deploy the app from GitHub.

## What's included

- `main.py` — dataset loading, TF-IDF features, Logistic Regression training, evaluation, and report-image creation.
- `app.py` — color-coded Streamlit text classifier.
- `notebooks/train_in_colab.ipynb` — trains a model in Colab and uploads the `.pkl` model and reports to this repository.
- `data/sentiment_sample.csv` — small offline three-class demonstration dataset.

Sentiment140 and Keras IMDB are both binary datasets, so their models predict positive or negative. The app supports neutral too when it is trained on a dataset that contains neutral examples, such as the included sample CSV.

## Train in Google Colab and save artifacts to GitHub

1. Create a GitHub personal access token with **Contents: Read and write** access to `isravel-eng/SentimentalAnalyst`.
2. Open the Colab notebook: [`notebooks/train_in_colab.ipynb`](https://colab.research.google.com/github/isravel-eng/SentimentalAnalyst/blob/main/notebooks/train_in_colab.ipynb).
3. In Colab, open **Secrets** in the left sidebar, create a secret named `GITHUB_TOKEN`, paste the token, and allow the notebook to access it.
4. Run the notebook cells in order. It installs the packages, downloads Sentiment140, trains using a balanced sample of up to 100,000 tweets, evaluates the model, and uploads these files to the repository:
   - `models/sentiment_model.pkl`
   - `reports/evaluation.txt`
   - `reports/evaluation_report.png`

The token is read from Colab Secrets and is never saved in the notebook or repository. The first Sentiment140 download is about 80 MB. Change the training command in the notebook to `--dataset imdb` if you want the Keras movie reviews instead; the notebook includes the needed installation step.

## Run the app locally

After the Colab notebook uploads `models/sentiment_model.pkl`, clone the repository or pull the latest changes, then run:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The app shows the predicted sentiment in green (positive), red (negative), or amber (neutral), along with confidence and each class score.

## Deploy on Streamlit Community Cloud

1. Finish the Colab run so `models/sentiment_model.pkl` is present on the repository's `main` branch.
2. In Streamlit Community Cloud, create a new app and choose this repository, the `main` branch, and `app.py` as the app file.
3. Deploy. The app installs the packages from `requirements.txt` and loads the model stored in `models/`.

If the model has not been uploaded yet, the app displays a message directing you to run the Colab notebook first.

## Train from a terminal instead

```bash
python -m pip install -r requirements.txt
python main.py train --dataset sentiment140 --max-samples 100000
```

Choose `--dataset imdb` after installing `requirements-imdb.txt`, or use `--dataset sample` for the offline three-class demo. Training writes a model, readable evaluation text, and a PNG report containing per-class scores and the confusion matrix.

