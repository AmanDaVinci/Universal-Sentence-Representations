config = {
    "exp_name": "baseline",
    "epochs": 20,
    "encoder": "EmbeddingEncoder",
    "batch_size": 128,
    "learning_rate": 1e-3,
    "seed": 42,
    "debug": False,
    # "device": 'cpu',
    "device": 'cuda',
    "num_workers": 4,
    "valid_freq": 1000,
    "save_freq": 2000,
}