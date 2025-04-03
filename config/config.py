from dataclasses import dataclass

@dataclass
class ModelConfig:
    image_size: int
    use_batchnorm: bool
    load_from_checkpoint: bool

    def __post_init__(self):
        if self.image_size <= 0:
            raise ValueError(f"image_size must be greater than 0, got {self.image_size}")

@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    num_epochs: int
    checkpoint_name: str
    checkpoint_dir: str
    log_dir: str

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError(f"Batch size must be greater or equal to 1, got {self.batch_size}")
        if self.learning_rate > 1 or self.learning_rate < 1e-7 :
            raise ValueError(f"Learning rate must be between 0 and 1, got {self.learning_rate}")
        if self.num_epochs < 1:
            raise ValueError(f"Number of epochs must be greater or equal to 1, got {self.num_epochs}")
        
@dataclass
class DatasetConfig:
    labelmap_path: str
    colormap_path: str
    train_image_path: str
    train_mask_path: str
    test_image_path: str
    test_mask_path: str
    test_image: str


@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    dataset: DatasetConfig

