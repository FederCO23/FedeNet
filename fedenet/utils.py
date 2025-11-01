def assert_input_shape(x, h=540, w=960):
    assert x.ndim == 4 and x.shape[2:] == (h, w), f"Expected (B,C,{h},{w}); got {tuple(x.shape)}"
