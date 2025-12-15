import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import hydra
from src.trainer import Trainer


@hydra.main(config_path="conf", config_name="food11_exp7", version_base="1.3")
def main(cfg: DictConfig):
    transforms = instantiate(cfg.transforms)
    test_dataset = instantiate(cfg.dataset, mode="test", transforms=transforms.test)
    test_loader = instantiate(cfg.dataloader, dataset=test_dataset, shuffle=False)

    print(cfg.experiment.best_model_save_path)
    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(cfg.experiment.best_model_save_path, map_location='cpu'))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    metrics = instantiate(cfg.metrics, device=device)._metrics
    
    trainer = Trainer(
        model=model,
        criterion=criterion,
        device=device,
        metrics=metrics,
        log_tb=False
    )
    
    test_loss, test_metrics = trainer.test(test_loader)
    

if __name__ == "__main__":
    main()