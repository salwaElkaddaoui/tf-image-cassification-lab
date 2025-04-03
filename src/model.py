import os, sys
import json
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable
import hydra
from config.config import Config

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()


@register_keras_serializable(package="CustomLayers")
class ClassificationHead(tf.keras.layers.Layer):
    def __init__(self, num_filters_dense1, num_classes, use_batchnorm=True, **kwargs):
        super().__init__(**kwargs)
        self.num_filters_dense1 = num_filters_dense1
        self.num_classes = num_classes
        self.use_batchnorm = use_batchnorm
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=self.num_filters_dense1, kernel_initializer='he_normal', bias_initializer='zeros')
        self.dense2 = tf.keras.layers.Dense(units=self.num_classes, activation='softmax', kernel_initializer='he_normal', bias_initializer='zeros')
        self.bn = tf.keras.layers.BatchNormalization()

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_filters_dense1': self.num_filters_dense1,
            'num_classes': self.num_classes,
            'use_batchnorm': self.use_batchnorm
        })
        return config

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        if self.use_batchnorm:
            x = self.bn(x, training=training)
        x = self.dense2(x)
        return x

@register_keras_serializable(package="CustomModel")    
class Classifier(tf.keras.Model):
    def __init__(self, input_shape, labelmap_path, use_batchnorm=True, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.use_batchnorm = use_batchnorm
        self.num_classes = self._load_num_classes(labelmap_path)

        # Define input tensor explicitly
        inputs = tf.keras.Input(shape=input_shape)

        # Backbone
        base_model = tf.keras.applications.ResNet101(include_top=False, weights='imagenet', input_shape=self.input_shape)
        backbone_input = base_model.input
        backbone_output = base_model.get_layer('conv4_block23_out').output
        self.backbone = tf.keras.Model(inputs=backbone_input, outputs=backbone_output, name="backbone")
        # Head
        self.head = ClassificationHead(num_filters_dense1=1024, num_classes=self.num_classes, use_batchnorm=self.use_batchnorm)

        # Call the layers to track the model
        # backbone_outs = self.backbone(inputs, training=False)
        # outputs = self.head(backbone_outs, training=False)

        # Define the complete model
        # self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Classifier")
        self.freeze_backbone()

    def _load_num_classes(self, labelmap_path):
        with open(labelmap_path, "r") as f:
            labelmap = json.load(f)
        labelmap = {int(k): v for k, v in labelmap.items()}
        return len(labelmap)

    def freeze_backbone(self):
        for layer in self.backbone.layers:
            layer.trainable = False
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'use_batchnorm': self.use_batchnorm
        })
        return config

    def call(self, inputs, training=False):
        backbone_outs = self.backbone(inputs, training=training)
        return self.head(backbone_outs, training=training)
        # return self.model(inputs, training=training)
    
    