import tensorflow as tf
import parameters as params
from pathlib import Path
import keras
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import data_augmentation
import image


def load_img(filename: str) -> tf.Tensor:
    img = tf.io.read_file(filename)
    img = tf.io.decode_png(img, channels=3, dtype=tf.uint8)
    return tf.cast(img, tf.float32)


def load_samples(classes: int) -> dict[str, tuple[int, int, int, float]] | dict[str, tuple[int, int, float]]:
    samples: dict[str, tuple[int, int, int, float]] = {}
    img_dir = params.extracted_dataset_path
    if classes == 0:
        for img_path in (img_dir / "left").glob("*.[jp][pn]g"):
            stem = img_path.stem.split(" ")
            distance = int(stem[1])
            samples[str(img_path)] = (1.0, 0.0, 0.0, distance / 1000000)
        print(f"found {len(samples)} left samples")

        for img_path in (img_dir / "segmentation" / "left").glob("*.[jp][pn]g"):
            stem = img_path.stem.split(" ")
            distance = int(stem[1])
            samples[str(img_path)] = (1.0, 0.0, 0.0, distance / 1000000)
        print(f"found {len(samples)} segmentated left samples")

    if classes == 1:
        for img_path in (img_dir / "right").glob("*.[jp][pn]g"):
            stem = img_path.stem.split(" ")
            distance = int(stem[1])
            samples[str(img_path)] = (0.0, 1.0, 0.0, distance / 1000000)
        print(f"found {len(samples)} right samples")

        for img_path in (img_dir / "segmentation" / "right").glob("*.[jp][pn]g"):
            stem = img_path.stem.split(" ")
            distance = int(stem[1])
            samples[str(img_path)] = (0.0, 1.0, 0.0, distance / 1000000)
        print(f"found {len(samples)} segmentated right samples")

    if classes == 2:
        for img_path in (img_dir / "none_kick").glob("*.[jp][pn]g"):
            stem = img_path.stem.split(" ")
            distance = int(stem[1])
            samples[str(img_path)] = (0.0, 0.0, 1.0, distance / 1000000)
        print(f"found {len(samples)} none_kick samples")

        for img_path in (img_dir / "segmentation" / "none").glob("*.[jp][pn]g"):
            stem = img_path.stem.split(" ")
            distance = int(stem[1])
            samples[str(img_path)] = (0.0, 0.0, 1.0, distance / 1000000)
        print(f"found {len(samples)} segmentated none_kick samples")

    if classes == 3:
        for img_path in (img_dir / "ready").glob("*.[jp][pn]g"):
            stem = img_path.stem.split(" ")
            distance = int(stem[1])
            samples[str(img_path)] = (1.0, 0.0, distance / 1000000)
        print(f"found {len(samples)} ready samples")

    if classes == 4:
        for img_path in (img_dir / "none_ready").glob("*.[jp][pn]g"):
            stem = img_path.stem.split(" ")
            distance = int(stem[1])
            samples[str(img_path)] = (0.0, 1.0, distance / 1000000)
        print(f"found {len(samples)} none_ready samples")

    if len(samples) == 0:
        raise FileNotFoundError

    return samples


def flip_image_and_label_kick_in(
    img_and_distance: tuple[tf.Tensor, tf.Tensor], label: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    img, distance = img_and_distance
    flipped_image = tf.image.flip_left_right(img)

    # Labels need to change: left <-> right
    new_label = tf.cond(
        tf.equal(label[0], 1),  # if "left"
        lambda: tf.constant([0, 1, 0], dtype=tf.float32),  # change it as "right"
        lambda: tf.cond(
            tf.equal(label[1], 1),  # if "right"
            lambda: tf.constant([1, 0, 0], dtype=tf.float32),  # change it as "left"
            lambda: label,  # "None" bleibt gleich
        ),
    )
    return (flipped_image, distance), new_label


def flip_image_and_label_standby_to_ready(
    img_and_distance: tuple[tf.Tensor, tf.Tensor], label: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    img, distance = img_and_distance
    flipped_image = tf.image.flip_left_right(img)

    return (flipped_image, distance), label


def get_tf_dataset_kick_in(
    classes: int,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    samples = load_samples(classes)

    imgs = tf.data.Dataset.from_tensor_slices(samples.keys())

    imgs_label = tf.data.Dataset.from_tensor_slices([v[:3] for v in samples.values()])  # Label
    imgs_distance = tf.data.Dataset.from_tensor_slices([v[3] for v in samples.values()])  # Distance

    imgs_data = imgs.map(
        partial(load_img),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = tf.data.Dataset.zip((imgs_data, imgs_distance), imgs_label)

    flipped_dataset = dataset.map(
        lambda img_and_distance, label: flip_image_and_label_kick_in(img_and_distance, label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.concatenate(flipped_dataset)

    return keras.utils.split_dataset(
        dataset,
        left_size=0.8,
        right_size=0.2,
        seed=seed,
    )


def get_tf_dataset_standby_to_ready(
    classes: int,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    samples = load_samples(classes)

    imgs = tf.data.Dataset.from_tensor_slices(samples.keys())

    imgs_label = tf.data.Dataset.from_tensor_slices([v[:2] for v in samples.values()])  # Label
    imgs_distance = tf.data.Dataset.from_tensor_slices([v[2] for v in samples.values()])  # Distance

    imgs_data = imgs.map(
        partial(load_img),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = tf.data.Dataset.zip((imgs_data, imgs_distance), imgs_label)
    flipped_dataset = dataset.map(
        lambda img_and_distance, label: flip_image_and_label_standby_to_ready(
            img_and_distance, label
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.concatenate(flipped_dataset)

    return keras.utils.split_dataset(
        dataset,
        left_size=0.8,
        right_size=0.2,
        seed=seed,
    )


def concat_datasets(
    datasets: list[tuple[tf.data.Dataset, tf.data.Dataset]],
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    if len(datasets) == 1:
        return datasets[0]
    ds_data = datasets[0][0]
    ds_label = datasets[0][1]
    return concat_datasets(
        [(ds_data.concatenate(datasets[1][0]), ds_label.concatenate(datasets[1][1]))] + datasets[2:]
    )


def get_all_tf_datasets_kick_in() -> tuple[tf.data.Dataset, tf.data.Dataset]:
    left = get_tf_dataset_kick_in(params.left, params.seed_train)
    right = get_tf_dataset_kick_in(params.right, params.seed_train)
    none = get_tf_dataset_kick_in(params.none, params.seed_train)

    ds = concat_datasets([left, right])

    print(f"found {len(ds[0]) + len(ds[1])} total samples")
    return (ds[0], ds[1])


def get_all_tf_datasets_standby_to_ready() -> tuple[tf.data.Dataset, tf.data.Dataset]:
    referee = get_tf_dataset_standby_to_ready(params.standby_to_ready, params.seed_train)
    none_initial = get_tf_dataset_standby_to_ready(params.none_ready, params.seed_train)

    ds = concat_datasets([referee, none_initial])

    print(f"found {len(ds[0]) + len(ds[1])} total samples")
    return (ds[0], ds[1])


def built_extracted_dataset(dataset:tuple[tf.data.Dataset, tf.data.Dataset]) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    ds_train, ds_valid = dataset
    ds_train = ds_train.shuffle(
        ds_train.cardinality(), seed=params.seed_train, reshuffle_each_iteration=True,
    )

    augmentation = data_augmentation.DataAugmentation()
    ds_train = ds_train.map(
        augmentation.augment, deterministic=True, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)


    ds_train = ds_train.map(
        lambda x, y: ((image.rgb2ycbcr(x[0]), x[1]), y), num_parallel_calls=tf.data.AUTOTUNE
    )

    ds_valid = ds_valid.map(
        lambda x, y: ((image.rgb2ycbcr(x[0]), x[1]), y), num_parallel_calls=tf.data.AUTOTUNE
    )
    if False: # could be useful for debugging
        for x, y in ds_train:
            x_rgb = x[0]
            for i in range(x_rgb.shape[0]):
                x_ycbcr = image.rgb2ycbcr(x_rgb[i])

                plt.imshow(x_ycbcr.numpy().astype(np.uint8))
                plt.show()

    return ds_train, ds_valid
