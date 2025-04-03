import json
import pathlib
import numpy as np
from sklearn.metrics import confusion_matrix,  accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Union
import tensorflow as tf
import os

class Evaluator:
    def __init__(self, labelmap_path: Union[str, pathlib.Path], log_dir: str, threshold: int=0.5):
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

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        
        y_pred_top1 = y_pred.argmax(axis=1)
        acc = accuracy_score(y_true, y_pred_top1)
        cm = confusion_matrix(y_true, y_pred_top1, normalize='true')
        global_precision = precision_score(y_true, y_pred)
        global_recall = recall_score(y_true, y_pred)
        
        per_class_metrics = {}
        for class_idx, class_name in self.labelmap.items():
            y_true_class = (y_true == class_idx).astype(int)
            y_pred_class = (y_pred_top1 == class_idx).astype(int)
            per_class_metrics[class_name] = {
                "accuracy": accuracy_score(y_true_class, y_pred_class),
                "precision": precision_score(y_true_class, y_pred_class, zero_division=0),
                "recall": recall_score(y_true_class, y_pred_class, zero_division=0)
            }

        return {
            "global": {"accuracy": acc, "precision": global_precision, "recall": global_recall, "confusion_matrix": cm},
            "per_class": per_class_metrics
        }

    def save_results(self, metrics: Dict[str, Any]) -> None:
        with open(os.path.join(self.log_dir,"metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        self._plot_confusion_matrix(np.array(metrics["global"]["confusion_matrix"]),save_path=os.path.join(self.log_dir, "confusion_matrix.png"))

    
    def _plot_confusion_matrix(self, cm: np.ndarray, save_path: pathlib.Path) -> None:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt=".2f", xticklabels=self.class_names, yticklabels=self.class_names, cmap="Blues", vmin=0, vmax=1)
        plt.title("Normalized Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def _create_writers(self):
        tb_writers = {}
        for class_name in self.labelmap.values():
            folder_path = os.path.join(self.log_dir, class_name)
            if not os.path.exists(folder_path): os.makedirs(folder_path)
            tb_writers[class_name] = tf.summary.create_file_writer(folder_path)
        
        folder_path = os.path.join(self.log_dir, "global")
        if not os.path.exists(folder_path): os.makedirs(folder_path)
        tb_writers["global"] = tf.summary.create_file_writer(folder_path)
        return tb_writers

    def log_to_tensorboard(self, metrics, epoch, name):
        with self.tb_writers['global']['accuracy'].as_default():
            tf.summary.scalar(name=f"{name}/global_accuracy", data=metrics["global"]["accuracy"], step=epoch)
        with self.tb_writers['global']['precision'].as_default():
            tf.summary.scalar(name=f"{name}/global_precision", data=metrics["global"]["precision"], step=epoch)
        with self.tb_writers['global']['recall'].as_default():
            tf.summary.scalar(name=f"{name}/global_recall", data=metrics["global"]["recall"], step=epoch)

        for class_name in self.evaluator.labelmap.values():
            with self.tb_writers["per_class"][class_name]["accuracy"].as_default():
                tf.summary.scalar(name=f"{name}/accuracy", data=metrics["per_class"][class_name]["accuracy"], step=epoch)
            with self.tb_writers["per_class"][class_name]["precision"].as_default():
                tf.summary.scalar(name=f"{name}/precision", data=metrics["per_class"][class_name]["precision"], step=epoch)
            with self.tb_writers["per_class"][class_name]["recall"].as_default():
                tf.summary.scalar(name=f"{name}/recall", data=metrics["per_class"][class_name]["recall"], step=epoch)


class EvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self, 
                 evaluator: Evaluator, 
                 train_data: tf.data.Dataset,
                 val_data: tf.data.Dataset
                 ):
        self.evaluator = evaluator
        self.train_data = train_data
        self.val_data = val_data
        
    def on_epoch_end(self, epoch, logs=None):    
        def eval(dataset: tf.data.Dataset, name):
            y_pred =[]
            y_true=[]
            for (images, labels) in dataset:
                y_pred.extend(self.model.predict(images, verbose=0))
                y_true.extend(labels)
            metrics = self.evaluator.compute_metrics(np.array(y_true), np.array(y_pred))
            self.evaluator.log_to_tensorboard(metrics, epoch, name)
        eval(self.train_data, "train")
        eval(self.val_data, "val")
