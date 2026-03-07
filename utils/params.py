def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def print_model_parameters(model):

    total = count_parameters(model)

    print(f"Total trainable parameters: {total}")
