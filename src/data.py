import tensorflow as tf
from typing import Union
import pdb
import json
from typing import Tuple

class DataLoader:
    def __init__(self, img_size: int, batch_size: int, labelmap_path: str):
        self.img_size = img_size
        self.batch_size = batch_size
        self.labelmap_path = labelmap_path

    def load_data(self, img_path: str)->Tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        
        parts = tf.strings.split(img_path, '/')
        label_key = parts[-3]
        label_idx = tf.py_function(
            lambda x: self.labelmap[x.numpy().decode('utf-8')],
            [label_key],
            tf.int32
        )
        label = tf.reshape(label_idx, ()) 
        return image, label
    
    def preprocess(self, image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.image.resize(image, [self.img_size, self.img_size], method='bilinear')
        image = tf.cast(image, tf.float32)/ 255.0  
        return image, label

    def augment(self, image: tf.Tensor, label: tf.Tensor)-> tuple[tf.Tensor, tf.Tensor]:
        choice = tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32)
        
        def rotate():
            return tf.image.rot90(image), label
        
        def shift():
            """
            shift by padding and cropping
            """
            pad_left = tf.random.uniform((), minval=0, maxval=50, dtype=tf.int32)
            pad_top = tf.random.uniform((), minval=0, maxval=50, dtype=tf.int32)
            
            if tf.random.uniform(()) > 0.5: #shift to the right 
                # 1. pad
                padded_image = tf.pad(image, [[0, 0], [pad_left, 0], [0, 0]], constant_values=0)
                # 2. crop
                return padded_image[:image.shape[0], :image.shape[1], :], label
                
            else: #shift to the bottom
                # 1. pad
                padded_image = tf.pad(image, [[pad_top, 0], [0, 0], [0, 0]], constant_values=0)
                # 2. crop
                return padded_image[:image.shape[0], :image.shape[1], :], label

        image, label = tf.switch_case(choice, branch_fns={0: rotate, 1: shift})
        return image, label


    def create_dataset(self, image_paths:str, training:bool=True) -> tf.data.Dataset:
        """ 
        Creates a TensorFlow Dataset from image file paths for training and evaluation.
        Args:
            image_paths: path of a .txt file containing the absolute path of images, one per line.
            training: Flag to apply image augmentation. Set to True for training, False for evaluation.
        Returns:
            tf.data.Dataset: a batched and preprocessed dataset, ready for training and evaluation.
        """
        image_paths = tf.data.TextLineDataset(image_paths)
        dataset = image_paths.map(self.load_data, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            dataset = dataset.map(self.augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            return dataset.shuffle(1000, reshuffle_each_iteration=True).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            return dataset.batch(self.batch_size).cache().prefetch(tf.data.AUTOTUNE)
        
        
    @property
    def labelmap(self):
        with open(self.labelmap_path, "r") as f:
            labelmap=json.load(f)
        return {v: int(k) for k,v in labelmap.items()}
    