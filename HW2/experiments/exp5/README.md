## Взяли новую архитектуру

### Параметры конфига

```
dataset:
  _target_: src.dataset.FoodDataset
  data_path: ./data/food11
transforms:
  _target_: src.transforms.Transforms_v2
  mean: (0.485, 0.456, 0.406)
  std: (0.229, 0.224, 0.225)
  resize: (224, 224)
dataloader:
  _target_: src.dataloader.BaseDataLoader
  batch_size: 64
  shuffle: true
  num_workers: 2
  pin_memory: true
train:
  num_epoch: 140
model:
  _target_: src.model.ResNetV2
  num_classes: 43
  dropout_rate: 0.3
load_dataset:
  data_path: ./data/food11
  download_path: ./data
  name: imbikramsaha/food11
metrics:
  _target_: src.metrics.ClassificationMetrics
  num_classes: 43
optimizer:
  _target_: src.optimizer.AdamWOptimizer
  lr: 0.0001
  weight_decay: 0.001
scheduler:
  _target_: src.scheduler.ReduceLROnPlateauScheduler
  mode: min
  factor: 0.5
  patience: 6
  threshold: 0.0001
  min_lr: 1.0e-07
seed: 42
experiment:
  name: exp5
  log_dir: experiments/${experiment.name}/
  tb_log_dir: runs/${experiment.name}
  best_model_save_path: experiments/${experiment.name}/models/best_model.pth
  last_model: None
```

## Описание эксперимента
- Число эпох: 95
- Архитектура: resnet_model_v2 (src/model/)
- Dropout: 0.3


Взял более сложную архитектуру (по два сверточных слоя 128 и 256 вместо одинх в прошлой архитектуре. Напоминает ResNet18 без последнего слоя и с одим сверточным слоем 64 вместо двух).

## Результаты

На тесте: 

Test Loss: 1.2657 | accuracy: 0.5690 | precision: 0.5737 | recall: 0.5690 | f1: 0.5641

На валидации:

Val Loss: 1.2300 | accuracy: 0.5850 | precision: 0.5881 | recall: 0.5839 | f1: 0.5758


## Вывод
Исходя из графиков обучения, модель явно недообучилась, нужно продолжать обучение.
