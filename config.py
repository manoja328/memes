from pathlib import Path

data_dir = Path.cwd() / "data"
img_path = data_dir / "img"
train_path = data_dir / "train.jsonl"
dev_path = data_dir / "dev.jsonl"
test_path = data_dir / "test.jsonl"

hparams = {

    # Required hparams
    "train_path": train_path,
    "dev_path": dev_path,
    "img_dir": data_dir,

    # Optional hparams
    "test_path": test_path,
    "embedding_dim": 150,
    "language_feature_dim": 300,
    "vision_feature_dim": 300,
    "fusion_output_size": 256,
    "output_path": "model-outputs",
    "dev_limit": None,
    "lr": 0.00005,
    "max_epochs": 10,
    "n_gpu": 1,
    "batch_size": 4,
    # allows us to "simulate" having larger batches
    "accumulate_grad_batches": 16,
    "early_stop_patience": 3,
}