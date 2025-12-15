## Дообучение лучшей модели из exp2

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
  num_workers: 2
  pin_memory: true
train:
  num_epoch: 60
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
  patience: 8
  threshold: 0.0001
  min_lr: 1.0e-07
seed: 42
experiment:
  name: exp4
  log_dir: experiments/${experiment.name}/
  tb_log_dir: runs/${experiment.name}
  best_model_save_path: experiments/${experiment.name}/models/best_model.pth
  last_model: experiments/exp2/models/best_model.pth
```

## Описание эксперимента
- Число эпох: 60 


Взяли лучшую модель с exp2, подгрузили в ту же архитектуру ее веса. Поменяли dropout 0.5 -> 0.3 и аугментации на более простые (Transforms_v2 -> Transforms_v1).

## Результаты

На тесте: 

Test Loss: 0.8031 | accuracy: 0.7540 | precision: 0.7566 | recall: 0.7540 | f1: 0.7544

На валидации:

Val Loss: 0.6710 | accuracy: 0.7933 | precision: 0.7942 | recall: 0.7923 | f1: 0.7927


Видим, что метрики на тесте чуть больше, чем в exp3 и exp2. Однако, исходя из графиков поведения лосса на трейне и валидации, можно сделать вывод что модель начала переобучаться.

## Вывод

Исходя из результатов последних двух экспериментов, можно сделать вывод, что из текущей архитектуры можно выжать еще 1-2% accuracity дообучением модели из exp2, взяв более сильные аугментации. С ограниченными вычислительными ресурсами тратить время на это не хочется, поэтому перейдем на другую архитектуру.
