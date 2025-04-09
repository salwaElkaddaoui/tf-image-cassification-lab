import json
import pathlib
import tensorflow as tf
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from typing import Dict, Any, Union


class Evaluator:
    def __init__(self, labelmap_path: Union[str, pathlib.Path], log_dir: str, threshold: float = 0.5):
        self.labelmap = self._load_labelmap(labelmap_path)
        self.class_names = [self.labelmap[i] for i in sorted(self.labelmap.keys())]
        self.log_dir = pathlib.Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.tb_writers = self._create_writers()

    def _load_labelmap(self, path: Union[str, pathlib.Path]) -> Dict[int, str]:
        with open(path) as f:
            labelmap = json.load(f)
        return {int(k): v for k, v in labelmap.items()}

    def compute_metrics(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> Dict[str, Any]:
        y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_true = tf.cast(y_true, tf.int32)

        num_classes = len(self.labelmap)

        confusion_matrix = tf.math.confusion_matrix(
            labels=y_true,
            predictions=y_pred_labels,
            num_classes=num_classes,
            dtype=tf.float32
        )
        normalized_cm = confusion_matrix / (tf.reduce_sum(confusion_matrix, axis=1, keepdims=True) + 1e-7)

        TP = tf.linalg.diag_part(confusion_matrix)
        FP = tf.reduce_sum(confusion_matrix, axis=0) - TP
        FN = tf.reduce_sum(confusion_matrix, axis=1) - TP
        class_counts = tf.reduce_sum(confusion_matrix, axis=1)

        precision = TP / (TP + FP + 1e-7)
        recall = TP / (TP + FN + 1e-7)
        accuracy = TP / (class_counts + 1e-7)

        result = {
            "confusion_matrix": normalized_cm.numpy().tolist(),
            "precision": {self.labelmap[i]: float(p) for i, p in enumerate(precision)},
            "recall": {self.labelmap[i]: float(r) for i, r in enumerate(recall)},
            "accuracy": {self.labelmap[i]: float(a) for i, a in enumerate(accuracy)},
        }

        # Compute means
        result["precision"]["mean"] = float(tf.reduce_mean(precision))
        result["recall"]["mean"] = float(tf.reduce_mean(recall))
        result["accuracy"]["mean"] = float(tf.reduce_mean(accuracy))
        return result

    def save_results(self, metrics: Dict[str, Any]) -> None:
        with open(os.path.join(self.log_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        self._plot_confusion_matrix(np.array(metrics["confusion_matrix"]),
                                     save_path=os.path.join(self.log_dir, "confusion_matrix.png"))

    def _plot_confusion_matrix(self, cm: np.ndarray, save_path: pathlib.Path) -> None:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt=".2f", xticklabels=self.class_names,
                    yticklabels=self.class_names, cmap="Blues", vmin=0, vmax=1)
        plt.title("Normalized Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def _create_writers(self):
        tb_writers = {}
        for item in list(self.labelmap.values()) + ["mean"]:
            folder_path = os.path.join(self.log_dir, item)
            os.makedirs(folder_path, exist_ok=True)
            tb_writers[item] = tf.summary.create_file_writer(folder_path)
        return tb_writers

    def log_to_tensorboard(self, metrics, epoch, name):
        for item in list(self.labelmap.values()) + ["mean"]:
            with self.tb_writers[item].as_default():
                tf.summary.scalar(name=f"{name}/accuracy", data=metrics["accuracy"][item], step=epoch)
                tf.summary.scalar(name=f"{name}/precision", data=metrics["precision"][item], step=epoch)
                tf.summary.scalar(name=f"{name}/recall", data=metrics["recall"][item], step=epoch)


class EvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self,
                 evaluator: Evaluator,
                 train_data: tf.data.Dataset,
                 val_data: tf.data.Dataset):
        self.evaluator = evaluator
        self.train_data = train_data
        self.val_data = val_data

    def on_epoch_end(self, epoch, logs=None):
        def eval(dataset: tf.data.Dataset, name):
            y_pred =[]
            y_true=[]
            for (images, labels) in dataset:
                y_pred.append(self.model.predict(images, verbose=0))
                y_true.append(labels)
            y_pred = tf.concat(y_pred, axis=0)
            y_true = tf.concat(y_true, axis=0)

            metrics = self.evaluator.compute_metrics(np.array(y_true), np.array(y_pred))
            self.evaluator.log_to_tensorboard(metrics, epoch, name)
        
        eval(self.train_data, "train")
        eval(self.val_data, "val")
