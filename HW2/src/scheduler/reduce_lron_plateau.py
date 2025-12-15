from torch.optim.lr_scheduler import ReduceLROnPlateau

class ReduceLROnPlateauScheduler(ReduceLROnPlateau):
    def __init__(
        self,
        optimizer,
        mode,
        factor,          
        patience,           
        threshold,     
        min_lr):
        super().__init__(optimizer=optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold, min_lr=min_lr)
