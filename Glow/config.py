def get_config():
    return {
        "hidden_channels":512,
        "K":32, #Number of layers per block
        "L":3, #Number of blocks
        "actnorm_scale": 1.0, #Act norm scale
        "flow_permutation":"invconv", #choices=["invconv", "shuffle", "reverse"]
        "flow_coupling":"affine",#["additive", "affine"]
        "LU_decomposed":True, #Train with LU decomposed 1x1 convs
        "learn_top":True, #train top layer (prior)
        "y_condition": True,#Train using class condition
        "y_weight":0.01, #Weight for class condition loss
        "max_grad_clip":0, #Max gradient value (clip above - for off)
        "max_grad_norm":0,#Max norm of gradient (clip above - 0 for off)
        "batch_size": 100,
        "epochs": 10, #250
        "lr": 5e-4,
        "warmup":5,
        "n_init_batches":8, #Number of batches to use for Act Norm initialisation
        "droprate":0.3,
        "momentum":0.9,
        "decay":0.0005,
        "output_dir":'./Glow/checkpoints',
        "saved_model": "",
        "saved_optimizer": "",
        "seed":0
    }