from __future__ import annotations

import albumentations as a
import numpy as np
import numpy.typing as npt
import tensorflow as tf


class DataAugmentation:
    def __init__(self):
        self.augmenter = a.Sequential(
            [
                a.GaussNoise(p=0.5, std_range=(0.01, 0.05)),
                a.Affine(
                    p=0.5,
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    scale=1.0,
                    rotate=0,
                ),
                a.OneOf(
                    [
                        a.SomeOf(
                            [
                                a.Illumination(
                                    p=0.5, mode="gaussian", intensity_range=(0.01, 0.2)
                                ),
                                a.MultiplicativeNoise(p=1, multiplier=(0.9, 1.2)),
                                a.RandomBrightnessContrast(
                                    p=0.5, brightness_limit=(-0.1, 0.1), ensure_safe_range=True
                                ),
                            ],
                            n=n,
                        )
                        for n in range(3)
                    ]
                ),
            ],
            p=0.5,
        )

    def augment_(
        self, image: npt.NDArray[np.float32], label: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        result = self.augmenter(image=image, label=label)
        return result["image"].astype(np.float32), result["label"].astype(np.float32)

    @tf.function
    def augment(self, image_and_distance: tuple[tf.Tensor, tf.Tensor], label: tf.Tensor) -> tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        image, distance = image_and_distance
        augmented_image, augmented_label = tf.numpy_function(
            self.augment_, inp=[image / 255.0, label], Tout=[tf.float32, tf.float32]
        )  # pyright: ignore[reportGeneralTypeIssues]
        augmented_image.set_shape(image.shape)
        augmented_image = augmented_image * 255.0
        augmented_image = tf.clip_by_value(augmented_image, 0.0, 255.0)
        augmented_label.set_shape(label.shape)
        return (augmented_image, distance), augmented_label
