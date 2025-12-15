import torch
import torch.nn as nn
from torch.utils.data import random_split, Subset
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import hydra
import logging
from src.dataset.load_dataset import download_dataset
from src.trainer import Trainer
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):

    seed = cfg.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    hydra_output_dir = os.getcwd()
    log_dir = os.path.join(hydra_output_dir, cfg.experiment.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, "train.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger("main")

    logger.info("\n" + "=" * 50)
    logger.info("⚙ Текущая конфигурация Hydra:\n%s", OmegaConf.to_yaml(cfg))
    logger.info("=" * 50 + "\n")

    # ------------------------------
    # 1️⃣ Конфигурация и скачивание датасета
    # ------------------------------

    _ = download_dataset(cfg.load_dataset.name, cfg.load_dataset.download_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # 2️⃣ Трансформации
    # ------------------------------
    transforms = instantiate(cfg.transforms)

    # ------------------------------
    # 3️⃣ Создание датасета и разделение на train/val
    # ------------------------------
    full_dataset = instantiate(cfg.dataset, mode="train", transforms=None)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_indices, val_indices = torch.utils.data.random_split(
        list(range(len(full_dataset))),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(1)
    )

    train_dataset = instantiate(cfg.dataset, mode="train", transforms=transforms.train)
    val_dataset = instantiate(cfg.dataset, mode="train", transforms=transforms.test)

    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    # ------------------------------
    # 4️⃣ DataLoader’ы
    # ------------------------------
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(seed)

    train_loader = instantiate(cfg.dataloader, dataset=train_dataset, generator=dataloader_generator)
    val_loader = instantiate(cfg.dataloader, dataset=val_dataset, generator=dataloader_generator)

    # ------------------------------
    # 5️⃣ Модель, функция потерь, оптимизатор
    # ------------------------------
    model = instantiate(cfg.model)
    # Раскоментировать, если хотим дообучить модель. Веса модели загрузятся из директории, указанной в конфиге
    # model.load_state_dict(torch.load(cfg.experiment.last_model, map_location='cpu'))
    criterion = nn.CrossEntropyLoss()
    optimizer = instantiate(cfg.optimizer, model = model)
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    # ------------------------------
    # 6️⃣ Метрики
    # ------------------------------
    metrics = instantiate(cfg.metrics, device=device)._metrics

    # ------------------------------
    # 7️⃣ Обучение
    # ------------------------------
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        metrics=metrics,
        scheduler=scheduler,
        log_dir=cfg.experiment.tb_log_dir,
        save_path = cfg.experiment.best_model_save_path
    )

    # trainer.train(num_epochs=cfg.train.num_epoch)
    train_losses, val_losses = trainer.train(
    num_epochs=cfg.train.num_epoch,    # Максимум эпох
    patience=12,       # Останавливаться после 10 эпох без улучшений
    min_epochs=80      # Начинать проверять early stopping только после 30 эпох
)
    # ------------------------------
    # 8️⃣ Тестирование
    # ------------------------------
    test_dataset = instantiate(cfg.dataset, mode="test", transforms=transforms.test)
    test_loader = instantiate(cfg.dataloader, dataset=test_dataset, shuffle = False)

    trainer.test(test_loader)


if __name__ == "__main__":
    main()
