from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class CosineAnnealingWarmRestartsScheduler(CosineAnnealingWarmRestarts):
    def __init__(
        self,
        optimizer,
        T_0,
        T_mult=1,
        eta_min=0,
        last_epoch=-1
    ):
        super().__init__(
            optimizer=optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
            last_epoch=last_epoch
        )