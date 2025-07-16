# keyneuron/config.py


ATTRIBUTION_CONFIG = {
    "gpt": {
        "transformer_layers_attr": "transformer.h",
        "input_ff_attr": "mlp.c_fc",
        "output_ff_attr": "mlp.c_proj.weight"
    },
    "glu_model" : {
    "transformer_layers_attr": "model.layers",
    "input_ff_attr": "mlp.up_proj",
    "output_ff_attr": "mlp.down_proj.weight"
    }
}


