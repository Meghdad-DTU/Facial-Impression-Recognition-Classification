from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    s3_bucket: str
    s3_key: str
    s3_secret_key: str
    object_key: Path
    local_data_file: Path
    local_train_file: Path
    local_val_file: Path
    local_test_file:Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path

    local_train_angry_dir: Path
    local_train_disgust_dir: Path
    local_train_fear_dir: Path
    local_train_happy_dir: Path
    local_train_sad_dir: Path
    local_train_surprise_dir: Path
    local_train_nuetral_dir: Path
    local_val_angry_dir: Path
    local_val_disgust_dir: Path
    local_val_fear_dir: Path
    local_val_happy_dir: Path
    local_val_sad_dir: Path
    local_val_surprise_dir: Path
    local_val_nuetral_dir: Path
    local_test_angry_dir: Path
    local_test_disgust_dir: Path
    local_test_fear_dir: Path
    local_test_happy_dir: Path
    local_test_sad_dir: Path
    local_test_surprise_dir: Path
    local_test_nuetral_dir: Path
    local_train_file: Path
    local_val_file: Path
    local_test_file: Path
    

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    model_path: Path
    updated_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int

@dataclass(frozen=True)
class PrepareCallbacksConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    ckeckpoint_model_filepath: Path
    patience: int

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    validation_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_imgage_size: list