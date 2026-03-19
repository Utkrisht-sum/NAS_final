def compute_parameters(model):

    return sum(p.numel() for p in model.parameters())

def estimate_flops(parameters):

    return parameters * 2