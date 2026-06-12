# SMS Spam Detection
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Modeling-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

An end-to-end machine learning project that classifies SMS messages as `spam` or `ham` using TF-IDF text features and multiple supervised learning models.
- a local dataset inside the repository
- a reusable training script
- automatic model comparison
- saved evaluation metrics and confusion matrix
- a simple prediction script
- a cleaner notebook workflow

## Project Highlights

- Cleans and preprocesses SMS text data
- Compares `MultinomialNB`, `LogisticRegression`, and `LinearSVC`
- Uses cross-validation to choose the best model
- Saves the trained model with `joblib`
- Generates evaluation reports for easier analysis

## Dataset

The project uses the SMS Spam Collection dataset.

- Source: [Kaggle - SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Local path used by this repo: `data/raw/spam.csv`

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Project Structure

```text
SMS-Spam-Detection/
|-- data/
|   `-- raw/
|       `-- spam.csv
|-- reports/
|   `-- metrics.json
|   `-- confusion_matrix.png
|-- models/
|   `-- spam_classifier.joblib
|-- predict.py
|-- train.py
|-- SPAM SMS DETECTION.ipynb
|-- requirements.txt
`-- README.md
```

## Installation

```bash
git clone https://github.com/nadeemahamad007/SMS-Spam-Detection.git
cd SMS-Spam-Detection
pip install -r requirements.txt
```

## Train the Model

Run the training pipeline:

```bash
python train.py
```

You can also provide a custom dataset path:

```bash
python train.py --data data/raw/spam.csv
```

This will:

- preprocess the dataset
- compare multiple models using cross-validation
- save the best model to `models/spam_classifier.joblib`
- save metrics to `reports/metrics.json`
- save the confusion matrix to `reports/confusion_matrix.png`

## Predict a New Message

After training, classify a new SMS message:

```bash
python predict.py "Congratulations! You have won a free ticket. Call now!"
```

Expected output:

```text
spam
```

## Model Workflow

1. Load the SMS dataset
2. Keep only the label and message columns
3. Clean the message text
4. Convert text into TF-IDF features
5. Train and compare multiple classifiers
6. Select the best model based on F1 score
7. Evaluate on a held-out test set

## Current Results

After running `python train.py` on the included dataset, the best-performing model was `LinearSVC`.

| Metric | Score |
|--------|-------|
| Accuracy | 98.74% |
| Precision | 98.56% |
| Recall | 91.95% |
| F1 Score | 95.14% |

Cross-validation F1 score comparison:

- `MultinomialNB`: 0.8736
- `LogisticRegression`: 0.9280
- `LinearSVC`: 0.9362

## Notebook

The notebook file `SPAM SMS DETECTION.ipynb` can be used for interactive exploration and demonstration. The main production workflow is now available in `train.py` and `predict.py`, which makes the project easier to run on any machine.

## Future Improvements

- deploy the model with Flask or Streamlit
- add hyperparameter tuning with GridSearchCV
- expose a web form for live spam prediction
- track experiments with MLflow

## Author

Nadeem Ahamad

Machine Learning project focused on SMS spam detection, message classification, spam pattern analysis, and model evaluation using Python by CodSoft Internship Project.

