config = {
    "exp_name": "bi_lstm_pool",
    "epochs": 20,
    "encoder": "BiLSTMPool",
    "batch_size": 128,
    "hidden_dim": 2048,
    "num_layers": 1,
    "learning_rate": 1e-3,
    "seed": 42,
    "debug": False,
    # "device": 'cpu',
    "device": 'cuda',
    "num_workers": 4,
    "valid_freq": 1000,
    "save_freq": 4000,
    "test_checkpoint": 'best-model.pt'
}
