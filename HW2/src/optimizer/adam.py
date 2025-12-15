from torch.optim import Adam

class AdamOptimizer(Adam):
    def __init__(self, model, lr=1e-3):
        super().__init__(
            model.parameters(), lr=lr
        )