## Дообучение прошлой лучшей модели

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
  num_epoch: 100
model:
  _target_: src.model.ResNet
  num_classes: 43
load_dataset:
  data_path: ./data/food11
  download_path: ./data
  name: imbikramsaha/food11
metrics:
  _target_: src.metrics.ClassificationMetrics
  num_classes: 43
optimizer:
  _target_: src.optimizer.AdamWOptimizer
  lr: 3.43e-05
  weight_decay: 0.001
scheduler:
  _target_: src.scheduler.ReduceLROnPlateauScheduler
  mode: min
  factor: 0.7
  patience: 8
  threshold: 0.0001
  min_lr: 1.0e-07
seed: 42
experiment:
  name: exp3
  log_dir: experiments/${experiment.name}/
  tb_log_dir: runs/${experiment.name}
  best_model_save_path: experiments/${experiment.name}/models/best_model.pth
  last_model: experiments/exp2/models/best_model.pth
```

## Описание эксперимента
- Число эпох: 29 (обучение было прервано по причине ограничения вычислительных ресурсов)

Была взята прошлая лучшая модель, подгружена из .pth в ту же архитектуру и дообучена с параметрами, соответствующими параметрам на момент остановки прошлого обучения.

## Результаты

На тесте: 

Test Loss: 0.7737 | accuracy: 0.7520 | precision: 0.7524 | recall: 0.7520 | f1: 0.7516

На валидации:

Val Loss: 0.6619 | accuracy: 0.7950 | precision: 0.7995 | recall: 0.7944 | f1: 0.7955


Можно видеть, что по сравнению с предыдущей моделью, метрики подросли.
