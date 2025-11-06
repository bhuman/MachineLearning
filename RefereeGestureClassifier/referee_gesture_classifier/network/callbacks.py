import keras
import parameters


model_early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=parameters.early_stopping_patience,
        restore_best_weights=True,
    )

model_learning_rate_callback = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=10,
    verbose=1,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=0.0001,
)

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    parameters.checkpoint_path,
    monitor="val_loss",
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
    initial_value_threshold=None,
)
