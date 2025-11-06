from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import tensorflow as tf
import parameters as params
import numpy as np
import shutil
import matplotlib.pyplot as plt
import math

def load_img(filename: Path) -> tf.Tensor:
    img = tf.io.read_file(str(filename))
    img = tf.io.decode_png(img, channels=3, dtype=tf.uint8)
    return img


def get_focal_length(width: float, opening_angle_deg: float) -> float:
    opening_angle_rad = math.radians(opening_angle_deg)
    return width / (2.0 * math.tan(opening_angle_rad / 2.0))


def get_size_by_distance(focal_length: float, size_in_reality: float, distance: float) -> float:
    return size_in_reality / distance * focal_length

def extract_patch(img: tf.Tensor, mode, distance: int) -> tf.Tensor:
    if mode == mode.KICK_IN:
        opening_angle_deg = 54.7
        width = 640
        size_in_reality = 2400
        size_by_distance = min(int(get_size_by_distance(get_focal_length(width, opening_angle_deg), size_in_reality, distance)), 480)
        size_by_distance = size_by_distance if size_by_distance % 2 == 0 else size_by_distance + 1
        img = tf.image.crop_to_bounding_box(
            img,
            0,
            (640 // 2) - (size_by_distance // 2),
            size_by_distance,
            size_by_distance,
        )
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(
            img,
            [256, 256],
            method="bilinear",
            preserve_aspect_ratio=False,
            antialias=False,
        )
    elif mode == mode.STANDBY_TO_READY:
        img = tf.image.crop_to_bounding_box(
            img,
            0,  # y offset
            (640 // 2) - (200 // 2),  # x offset
            256,  # height
            200   # width
        )
    return img


def load_raw_filepath_samples(folder_name) -> list[str]:
    (params.general_img_path / "left").mkdir(parents=True, exist_ok=True)
    (params.general_img_path / "right").mkdir(parents=True, exist_ok=True)
    (params.general_img_path / "none_kick").mkdir(parents=True, exist_ok=True)
    (params.general_img_path / "ready").mkdir(parents=True, exist_ok=True)
    (params.general_img_path / "none_ready").mkdir(parents=True, exist_ok=True)

    img_dir = params.general_img_path
    samples = [str(img_path) for img_path in (img_dir / folder_name).glob("*.[jp][pn]g")]
    print(f"found {len(samples)} {folder_name} samples")
    return samples

def load_raw_filepath_image() -> list[str]:
    samples = load_raw_filepath_samples("left")
    samples += load_raw_filepath_samples("right")
    samples += load_raw_filepath_samples("none_kick")
    samples += load_raw_filepath_samples("ready")
    samples += load_raw_filepath_samples("none_ready")

    if len(samples) == 0:
        raise FileNotFoundError

    return samples

def save_image_on_disk(filepath: Path, image: tf.Tensor) -> None:
    image = tf.image.convert_image_dtype(image, tf.uint8)
    image = tf.image.encode_png(image)
    new_path = Path(str(filepath).replace("dataset", "extracted_dataset", 1))
    tf.io.write_file(str(new_path), image)



def process_image(file_path: Path) -> None:
    img = load_img(file_path)
    class LocalTrainer(Enum):
        STANDBY_TO_READY = "standby_to_ready"
        KICK_IN = "kick_in"
    if file_path.parent.name == "ready" or file_path.parent.name == "none_ready":
        mode = LocalTrainer.STANDBY_TO_READY
    else:
        mode = LocalTrainer.KICK_IN

    stem = file_path.stem.split(" ")
    distance = stem[1]
    if ".png" in distance:
        distance = distance.split(".")[0]

    distance = int(distance)

    img = extract_patch(img, mode, distance)
    try:
        zeitstempel = int(stem[2])
    except IndexError:
        zeitstempel = 1000
    except ValueError:
        zeitstempel = str(stem[2])
    normalized_distance = (distance - params.normalize_min_distance) / (params.normalize_max_distance - params.normalize_min_distance)

    new_file_name = f"{stem[0]} {int(normalized_distance * 1000000):06d} {zeitstempel}.png"
    new_file_path = file_path.with_name(new_file_name)
    save_image_on_disk(new_file_path, img)

def save_extracted_dataset_on_disk() -> None:
    file_paths = load_raw_filepath_image()
    print("Saving Images: ")
    with ThreadPoolExecutor() as executor:
        executor.map(lambda file_path: process_image(Path(file_path)), file_paths)

    print("Done")

if __name__ == "__main__":
    params.extracted_dataset_path.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(params.extracted_dataset_path / "left", ignore_errors=True)
    shutil.rmtree(params.extracted_dataset_path / "right", ignore_errors=True)
    shutil.rmtree(params.extracted_dataset_path / "none_kick", ignore_errors=True)
    shutil.rmtree(params.extracted_dataset_path / "ready", ignore_errors=True)
    shutil.rmtree(params.extracted_dataset_path / "none_ready", ignore_errors=True)
    save_extracted_dataset_on_disk()
