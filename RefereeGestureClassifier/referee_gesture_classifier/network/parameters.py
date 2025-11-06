from __future__ import annotations
from pathlib import Path

import datetime
import tensorflow as tf


# Model parameters
epochs = 100

img_height = 480
img_width = 640

upper_camera_solution = (480, 640)

batch_size = 32
seed_train = 42

normalize_min_distance = 1000
normalize_max_distance = 9000

# Kick in resolution
extracted_img_height = 340
extracted_img_height_offset = 0
extracted_img_width = 340

scaled_img_height = 256
scaled_img_width = 256


# Label indices
left = 0
right = 1
none = 2

standby_to_ready = 3 # this is the class index
none_ready = 4 # this is the class index

standby_to_ready_index = 0
none_standby_to_ready_index = 1


# Callbacks
early_stopping_patience = 30

# Path parameters
module_path = Path().resolve()
general_img_path = module_path / "referee_gesture_classifier"/ "dataset"
extracted_dataset_path = module_path / "referee_gesture_classifier" / "extracted_dataset"
save_path = module_path / "models"

model_save_path = Path(
    module_path
    / "models"
    / datetime.datetime.now(datetime.timezone.utc).strftime("%d.%m.%Y-%H_%M_%S")
)
checkpoint_path = (
    module_path
    / "models"
    / "backup"
    / datetime.datetime.now(datetime.timezone.utc).strftime("%d.%m.%Y-%H_%M_%S")
    / "{epoch:02d}-{val_loss:.4f}.keras"
)

log_path = (
    module_path
    / "models"
    / "logs"
    / datetime.datetime.now(datetime.timezone.utc).strftime("%d.%m.%Y-%H_%M_%S")
)


