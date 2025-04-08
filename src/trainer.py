import tensorflow as tf
import numpy as np
from evaluator import Evaluator, EvaluationCallback
from model import Classifier
import hydra
from config.config import Config
import os
from data import DataLoader
import datetime

class Trainer:
    def __init__(self, model, optimizer, callbacks):
        self.model = model
        self.callbacks = callbacks
        self.optimizer = optimizer

    def train(self, train_data,  val_data, epochs):
        self.model.compile(optimizer = self.optimizer, loss='SparseCategoricalCrossentropy')
        self.model.fit(train_data, validation_data=val_data, 
                      epochs=epochs, callbacks=[self.callbacks]
                      )
    
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: Config):
    #create logdir
    log_dir = os.path.join(cfg.training.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(log_dir): os.makedirs(log_dir)

    # load model
    if cfg.model.load_from_checkpoint:
        model = tf.keras.models.load_model(os.path.join(cfg.training.checkpoint_path, cfg.training.checkpoint_name))
    else:
        model = Classifier(input_shape=(cfg.model.image_size, cfg.model.image_size, 3), 
                        labelmap_path=cfg.dataset.labelmap_path, 
                        use_batchnorm=cfg.model.use_batchnorm)

    
    #instantiate evaluator
    evaluator = Evaluator(labelmap_path=cfg.dataset.labelmap_path, log_dir=log_dir)

    #load train and val data
    data_processor = DataLoader(
        img_size=cfg.model.image_size, 
        batch_size=cfg.training.batch_size, 
        labelmap_path=cfg.dataset.labelmap_path
    )
    train_dataset = data_processor.create_dataset(
        image_paths=cfg.dataset.train_image_path, 
        training=True #for image augmentation
    )
    val_dataset = data_processor.create_dataset(
        image_paths=cfg.dataset.test_image_path, 
        training=False
    )

    #callbacks definition 
    evalcallback = EvaluationCallback(evaluator, train_dataset, val_dataset)

    #instantiate trainer
    trainer = Trainer(model,  optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.training.learning_rate), callbacks=[evalcallback] )
    trainer.train(train_dataset, val_dataset, cfg.training.num_epochs)

if __name__=='__main__':
    main()