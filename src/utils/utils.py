from model.transformer_model import Model

def countTrainableParameters(model: Model) -> int:
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params
