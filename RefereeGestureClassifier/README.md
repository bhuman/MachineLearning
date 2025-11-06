# RefereeGestureClassifier

## RefereeGesture Trainer
You need a Python 3.12+ environment with the following packages installed:

- albumentations=2.0.5
- keras=3.8.0
- matplotlib=3.10.1
- tensorflow=2.18.0
- tf2onnx=1.16.1
- pathlib=1.0.1

Afterwards, place a dataset in the `datasets` folder.

Run the generate_dataset.py script to extract patches from the images. It will be automatically saved in the `extracted_datasets` folder.

Then, run the referee_gesture_trainer.py script to train a model. The trained model will be saved in the `models` folder.


### Project Organization
```
│
├── .gitignore          <- Ignore everything specific for this project.
├── .pre-commit-config.yaml  <- Pre commit checks to ensure consistent code style and to avoid
│                               common errors.
├── README.md           <- The top-level README for developers using this project.
├── data
│   ├── processed       <- The final, canonical data sets for modeling.
│   └── raw             <- The original, immutable data dump. For example DVC, ImageTagger or logs.
│
│
├── referee_gesture_classifier        <- Source code for use in this project.
│   │
│   │
│   ├── network                      <- Scripts to train models and then use trained models to make
│   |   │                                  predictions (a folder for every experiment).
│   |   ├── build_dataset.py                <- This will generate a tensorflow dataset
│   |   ├── callbacks.py                    <- Callbacks for training
│   |   ├── data_augmentation.py            <- Data augmentation for the training dataset
│   |   ├── generate_dataset.py             <- Extract patches from raw images and save them in extracted_dataset
│   |   ├── image.py                        <- Image processing functions (for example converting RGB to YUV)
│   |   ├── metrics.py                      <- Custom metrics for the models
│   |   ├── model_saver.py                  <- Save model in h5 and onnx format
│   |   ├── models.py                       <- Model architectures
│   |   ├── parameters.py                   <- All key parameters for this module
│   |   └── referee_gesture_trainer.py      <- Main training script
│   |
│   ├── datasets                            <- Place the dataset here
│   |
│   └── extracted_datasets                   <- generate_dataset.py will save extracted dataset here
│
├── models              <- Trained and serialized models, model predictions, or model summaries and
│                         metadata (a folder for every experiment).
│
└── pyproject.toml      <- The python package metadata for this project.
```
