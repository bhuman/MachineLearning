from pathlib import Path
import tensorflow as tf
import tf2onnx
import os
import keras
import parameters as params



def save_model(model: keras.Model, suffix: str) -> None:
    model.save(params.model_save_path / f"{suffix}.h5")
    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature(model, True),
        output_path=str(params.model_save_path / f"{suffix}.onnx"),
    )


def convert_keras_to_onnx(
    h5_model_path: str | os.PathLike[str],
    onnx_dest_dir: str | os.PathLike[str] | None = None,
    filename: str | os.PathLike[str] | None = None,
    single_batch_input: bool = True,
) -> None:
    onnx_dest_dir = Path(onnx_dest_dir) if onnx_dest_dir else Path(h5_model_path).parent
    filename = h5_model_path if filename is None else filename
    model = tf.keras.models.load_model(str(h5_model_path), compile=False)
    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature(model, single_batch_input),
        output_path=str(onnx_dest_dir / Path(filename).with_suffix(".onnx").name),
    )


def input_signature(
    model: tf.keras.Model,
    single_batch_input: bool,
) -> list[tf.TensorSpec]:
    input_specs = []
    for spec in model.inputs:
        if single_batch_input:
            if spec.shape[0] is None:
                spec = tf.TensorSpec([1] + list(spec.shape[1:]), spec.dtype)
        input_specs.append(spec)
    return input_specs
