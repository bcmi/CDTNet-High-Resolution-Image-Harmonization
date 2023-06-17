from iharm.model.base import CDTNet


BMCONFIGS = {
    #txh
    'CDTNet': {
        'model': CDTNet,
        'params': {'depth': 4, 'ch': 32, 'image_fusion': True, 'attention_mid_k': 0.5,
                   'batchnorm_from': 2, 'attend_from': 2, 'n_lut': 4}
    },
}
