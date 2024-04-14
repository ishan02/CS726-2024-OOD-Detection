def get_config():
    return {
        "batch_size": 32,
        "epochs": 10,
        "learning_rate": 0.001,
        "layers":40,
        "widen_factor":2,
        "droprate":0.3,
        "momentum":0.9,
        "decay":0.0005,
        "save":"./checkpoints",
        "load":"./checkpoints/file_name",
        "test":True, #test only flag



    }
