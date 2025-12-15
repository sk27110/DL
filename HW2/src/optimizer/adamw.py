from torch.optim import AdamW

class AdamWOptimizer(AdamW):
    def __init__(self, model, lr = 1e-3, weight_decay = 1e-3):
        super().__init__(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )