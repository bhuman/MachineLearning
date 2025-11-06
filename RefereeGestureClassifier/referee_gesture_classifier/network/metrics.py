import tensorflow as tf
import keras



def calc_true_positives(y_true:tf.Tensor, y_pred:tf.Tensor, class_index:int, first_index:int, last_index:int) -> tf.Tensor:
    predicted_classes = tf.argmax(y_pred[..., first_index:last_index+1], axis=-1)
    true_classes = tf.argmax(y_true[..., first_index:last_index+1], axis=-1)
    sum = tf.math.reduce_sum(tf.cast(tf.math.logical_and((true_classes == class_index), (predicted_classes == class_index)), tf.float32), axis=-1)
    return tf.reshape(sum, [1])

def calc_false_positives(y_true:tf.Tensor, y_pred:tf.Tensor, class_index:int, first_index:int, last_index:int) -> tf.Tensor:
    predicted_classes = tf.argmax(y_pred[..., first_index:last_index+1], axis=-1)
    true_classes = tf.argmax(y_true[..., first_index:last_index+1], axis=-1)
    sum = tf.math.reduce_sum(tf.cast(tf.math.logical_and((true_classes != class_index), (predicted_classes == class_index)), tf.float32), axis=-1)
    return tf.reshape(sum, [1])

def calc_false_negatives(y_true:tf.Tensor, y_pred:tf.Tensor, class_index:int, first_index:int, last_index:int) -> tf.Tensor:
    predicted_classes = tf.argmax(y_pred[..., first_index:last_index+1], axis=-1)
    true_classes = tf.argmax(y_true[..., first_index:last_index+1], axis=-1)
    sum = tf.math.reduce_sum(tf.cast(tf.math.logical_and((true_classes == class_index), (predicted_classes != class_index)), tf.float32), axis=-1)
    return tf.reshape(sum, [1])

def calc_true_negatives(y_true:tf.Tensor, y_pred:tf.Tensor, class_index:int, first_index:int, last_index:int) -> tf.Tensor:
    predicted_classes = tf.argmax(y_pred[..., first_index:last_index+1], axis=-1)
    true_classes = tf.argmax(y_true[..., first_index:last_index+1], axis=-1)
    sum = tf.math.reduce_sum(tf.cast(tf.math.logical_and((true_classes != class_index), (predicted_classes != class_index)), tf.float32), axis=-1)
    return tf.reshape(sum, [1])



class Precision(keras.metrics.Precision):
    def __init__(self, class_index:int, first_index:int, last_index:int, name=None) -> None:
        super().__init__(name=name, thresholds=None)
        self.class_index = class_index
        self.first_index = first_index
        self.last_index = last_index

    def update_state(self, y_true:tf.Tensor, y_pred:tf.Tensor, sample_weight=None):
        self.true_positives.assign_add(calc_true_positives(y_true, y_pred, self.class_index, self.first_index, self.last_index))
        self.false_positives.assign_add(calc_false_positives(y_true, y_pred, self.class_index, self.first_index, self.last_index))



class Recall(keras.metrics.Recall):
    def __init__(self, class_index:int, first_index:int, last_index:int, name=None) -> None:
        super().__init__(name=name, thresholds=None)
        self.class_index = class_index
        self.first_index = first_index
        self.last_index = last_index

    def update_state(self, y_true:tf.Tensor, y_pred:tf.Tensor, sample_weight=None):
        self.true_positives.assign_add(calc_true_positives(y_true, y_pred, self.class_index, self.first_index, self.last_index))
        self.false_negatives.assign_add(calc_false_negatives(y_true, y_pred, self.class_index, self.first_index, self.last_index))




