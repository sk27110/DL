## Обучение ResNet на датасете food11

### Параметры конфига
```
dataset:
  _target_: src.dataset.FoodDataset
  data_path: ./data/food11
transforms:
  _target_: src.transforms.Transforms_v1
  mean: (0.485, 0.456, 0.406)
  std: (0.229, 0.224, 0.225)
  resize: (224, 224)
dataloader:
  _target_: src.dataloader.BaseDataLoader
  batch_size: 64
  shuffle: true
  num_workers: 4
  pin_memory: false
train:
  num_epoch: 140
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
  lr: 0.0001
  weight_decay: 0.001
scheduler:
  _target_: src.scheduler.ReduceLROnPlateauScheduler
  mode: min
  factor: 0.7
  patience: 5
  threshold: 0.0001
  min_lr: 1.0e-07
log_dir: experiments/exp1
seed: 42
```
## Описание эксперимента
- Число эпох: 76 (обучение было прервано по причине ограничения вычислительных ресурсов)
- Dropout модели: 0.2
- Оптимизатор: AdamW (параметры выше)
- Sheduler: ReduceLROnPlateauScheduler (параметры выше)
- Предобработка изображений и аугментации: transforms_v1 (transforms.py)

## Результаты

Из главной директории:

```
tensorboard --logdir=runs
```

Исходя из графика сравнения лосса на валидации и трейне, можно сделать вывод, что, начиная с 50-60 эпохи модель начинает переобучаться, так как лосс на валидации почти не падает, в отличие от лосса на трейне. 

### Лучшие результаты
Валидация: Val Loss: 0.8088 | accuracy: 0.7411 | precision: 0.7564 | recall: 0.7399 | f1: 0.7432