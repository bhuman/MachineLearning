from __future__ import annotations

from pathlib import Path

import keras
import matplotlib.pyplot as plt
import models
import numpy as np
import parameters as params
import tensorflow as tf
import callbacks
import model_saver
import metrics
import built_dataset


def loss_kick_in(
    y_true: tf.Tensor, y_pred: tf.Tensor
) -> tf.Tensor:
    return keras.losses.categorical_crossentropy(
        y_true[..., params.left : params.none + 1],
        y_pred[..., params.left : params.none + 1],
        from_logits=True
    )

def loss_standby_to_ready(
    y_true: tf.Tensor, y_pred: tf.Tensor
) -> tf.Tensor:
    return keras.losses.categorical_crossentropy(
        y_true[..., params.standby_to_ready_index : params.none_standby_to_ready_index + 1],
        y_pred[..., params.standby_to_ready_index : params.none_standby_to_ready_index + 1],
        from_logits=True
    )


def start_training_from_saved_dataset_kick_in(model):
    model.compile(
        loss=loss_kick_in,
        optimizer=keras.optimizers.Adam(),
        metrics=[
                metrics.Precision(params.left, params.left, params.none, name="precision_left"),
                metrics.Precision(params.right, params.left, params.none, name="precision_right"),
                metrics.Precision(params.none, params.left, params.none, name="precision_none"),
                metrics.Recall(params.left, params.left, params.none, name="recall_left"),
                metrics.Recall(params.right, params.left, params.none, name="recall_right"),
                metrics.Recall(params.none, params.left, params.none, name="recall_none"),
                ],
        run_eagerly=False,
        jit_compile=False,
    )

    ds_train, ds_valid = built_dataset.built_extracted_dataset(built_dataset.get_all_tf_datasets_kick_in())

    ds_train = ds_train.batch(params.batch_size)
    ds_valid = ds_valid.batch(params.batch_size)


    model.fit(
        x=ds_train,
        validation_data=ds_valid,
        batch_size=params.batch_size,
        epochs=params.epochs,
        callbacks=[
                   callbacks.model_early_stopping_callback,
                   callbacks.model_learning_rate_callback,
                   callbacks.model_checkpoint_callback,
                   ],
    )


def start_training_from_saved_dataset_standby_to_ready(model):
    model.compile(
        loss=loss_standby_to_ready,
        optimizer=keras.optimizers.Adam(),
        metrics=[
                metrics.Precision(params.standby_to_ready_index, params.standby_to_ready_index, params.none_standby_to_ready_index, name="precision_ready"),
                metrics.Precision(params.none_standby_to_ready_index, params.standby_to_ready_index, params.none_standby_to_ready_index, name="precision_none"),
                metrics.Recall(params.standby_to_ready_index, params.standby_to_ready_index, params.none_standby_to_ready_index, name="recall_ready"),
                metrics.Recall(params.none_standby_to_ready_index, params.standby_to_ready_index, params.none_standby_to_ready_index, name="recall_none"),
                ],
        run_eagerly=False,
        jit_compile=False,
    )

    ds_train, ds_valid = built_dataset.built_extracted_dataset(built_dataset.get_all_tf_datasets_standby_to_ready())

    ds_train = ds_train.batch(params.batch_size)
    ds_valid = ds_valid.batch(params.batch_size)


    model.fit(
        x=ds_train,
        validation_data=ds_valid,
        batch_size=params.batch_size,
        epochs=params.epochs,
        callbacks=[
                callbacks.model_early_stopping_callback,
                callbacks.model_learning_rate_callback,
                callbacks.model_checkpoint_callback,
                ],
        )



if __name__ == "__main__":
    model = models.model_standby_to_ready_without_softmax()
    #model = models.model_kick_in_without_softmax()
    model.summary()
    start_training_from_saved_dataset_standby_to_ready(model)
    #start_training_from_saved_dataset_kick_in(model)
    model_saver.save_model(model, f"ready_without_softmax_{params.seed_train}")
