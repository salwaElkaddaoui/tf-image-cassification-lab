import json
import pathlib
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Union
import tensorflow as tf
import os

class Evaluator:
    def __init__(self, labelmap_path: Union[str, pathlib.Path], log_dir: str, threshold: int=0.5):
        self.labelmap = self._load_labelmap(labelmap_path)
        self.class_names = [self.labelmap[i] for i in sorted(self.labelmap.keys())]
        self.class_indices = sorted(self.labelmap.keys())
        self.log_dir = pathlib.Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.tb_writers = self._create_writers()

    def _load_labelmap(self, path: Union[str, pathlib.Path]) -> Dict[int, str]:
        with open(path) as f:
            labelmap = json.load(f)
        return {int(k): v for k, v in labelmap.items()}

    def compute_precision(self, y_true, y_pred):
        TP = {}
        FP = {}
        precision = {}
        for class_idx, class_name in self.labelmap.items():
            TP[class_name] = np.sum((y_pred == class_idx)*(y_true == class_idx))
            FP[class_name] = np.sum((y_pred == class_idx)*(y_true != class_idx))
            precision[class_name] = TP[class_name]/(TP[class_name]+FP[class_name]+1e-7)
        precision["mean"] = np.mean(list(precision.values()))
        return precision

    def compute_recall(self, y_true, y_pred):
        TP={}
        FN={}
        recall={}
        for class_idx, class_name in self.labelmap.items():
            TP[class_name] = np.sum((y_pred==class_idx)*(y_true==class_idx))
            FN[class_name]=np.sum((y_pred!=class_idx)*(y_pred==class_idx))
            recall[class_name] = TP[class_name]/(TP[class_name]+FN[class_name]+1e-7)
        recall["mean"] = np.mean(list(recall.values()))
        return recall
    
    def compute_accuracy(self, y_true, y_pred):
        accuracy={}
        for class_idx, class_name in self.labelmap.items():
            accuracy[class_name] = np.sum((y_pred==class_idx)*(y_true==class_idx))/(np.sum(y_true==class_idx)+1e-7)
        accuracy["mean"] = np.mean(list(accuracy.values()))
        return accuracy

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:    
        y_pred_top1 = y_pred.argmax(axis=-1)
        cm = sklearn.metrics.confusion_matrix(y_true, y_pred_top1, normalize='true')
        accuracy = self.compute_accuracy(y_true, y_pred_top1)
        precision = self.compute_precision(y_true, y_pred_top1)
        recall = self.compute_recall(y_true, y_pred_top1)

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "confusion_matrix": cm}
 

    def save_results(self, metrics: Dict[str, Any]) -> None:
        with open(os.path.join(self.log_dir,"metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        self._plot_confusion_matrix(np.array(metrics["confusion_matrix"]),save_path=os.path.join(self.log_dir, "confusion_matrix.png"))

    
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
        for item in list(self.labelmap.values())+["mean"]:
            folder_path = os.path.join(self.log_dir, item)
            if not os.path.exists(folder_path): os.makedirs(folder_path)
            tb_writers[item] = tf.summary.create_file_writer(folder_path)
        return tb_writers

    def log_to_tensorboard(self, metrics, epoch, name):    
        for item in list(self.labelmap.values())+["mean"]:
            with self.tb_writers[item].as_default():
                tf.summary.scalar(name=f"{name}/accuracy", data=metrics["accuracy"][item], step=epoch)
                tf.summary.scalar(name=f"{name}/precision", data=metrics["precision"][item], step=epoch)
                tf.summary.scalar(name=f"{name}/recall", data=metrics["recall"][item], step=epoch)


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
