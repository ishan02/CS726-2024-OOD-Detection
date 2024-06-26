def get_config():
    return {
        "batch_size": 500,
        "epochs": 20,
        "learning_rate": 0.001,
        "layers":40,
        "widen_factor":2,
        "droprate":0.3,
        "momentum":0.9,
        "decay":0.0005,
        "folder":'./WRN/checkpoints',
        "save":"./WRN/checkpoints/epoch",
        "load_epoch":13,
        "preload": True
    }
