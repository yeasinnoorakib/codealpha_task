# Emotion Recognition from Speech (MFCC + CNN)

## Data format
Put WAV files like:
data/
  angry/*.wav
  happy/*.wav
  sad/*.wav
  neutral/*.wav

## Install
pip install -r requirements.txt

## Train
python -m src.train

## Evaluate
python -m src.evaluate
