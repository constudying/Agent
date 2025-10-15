
DEFAULT_DOWN_KWARGS = None
DOWN_DECODER_NETWORKS = {"None": None}

def register_down_decoder_network(target_class):
    assert target_class not in DOWN_DECODER_NETWORKS, \
        "Downstream decoder class {} already registered".format(target_class)
    DOWN_DECODER_NETWORKS[target_class.__name__] = target_class