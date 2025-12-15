# Disease Prediction from Medical Data (Sklearn)

## Data
Place your CSV at: data/dataset.csv
Set the target column name in src/config.py (default: target)

## Install
pip install -r requirements.txt

## Train
python -m src.train

## Evaluate (plots confusion matrix + ROC curve)
python -m src.evaluate

## Interpretability (feature importances if RandomForest)
python -m src.interpret
