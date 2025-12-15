## Дообучение модели из exp5

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
  num_epoch: 160
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
  lr: 5.0e-05
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
  name: exp6
  log_dir: experiments/${experiment.name}/
  tb_log_dir: runs/${experiment.name}
  best_model_save_path: experiments/${experiment.name}/models/best_model.pth
  last_model: experiments/exp5/models/best_model.pth
```


## Описание эксперимента
- Число эпох: 77 (прервано из-за нехватки ресурсов)
- Архитектура: resnet_model_v2 (src/model/)
- Dropout: 0.3
- Параметры взяты из окончания прошлого эксперимента

Взял прошлую лучщую модель их exp5 с параметрами конца обучения для продолжения обучения

## Результаты

На тесте:

Test Loss: 1.1470 | accuracy: 0.6230 | precision: 0.6384 | recall: 0.6230 | f1: 0.6245

На валидации:

Val Loss: 1.0858 | accuracy: 0.6444 | precision: 0.6486 | recall: 0.6440 | f1: 0.6429

## Вывод

На графиках и из логов видно, что на последних 10-15 эпохах улучшения практически не происходило даже с учетом уменьшения lr, в том числе и на трейне. Это может говорить о том, что модель уперлась в локальный минимум.