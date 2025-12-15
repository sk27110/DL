## Дообучение модели из exp5

### Параметры конфига
```
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
  lr: 0.0001
  weight_decay: 0.001
scheduler:
  _target_: src.scheduler.CosineAnnealingWarmRestartsScheduler
  T_0: 25
  T_mult: 2
  eta_min: 1.0e-06
  last_epoch: -1
seed: 42
experiment:
  name: exp7
  log_dir: experiments/${experiment.name}/
  tb_log_dir: runs/${experiment.name}
  best_model_save_path: experiments/${experiment.name}/models/best_model.pth
  last_model: experiments/exp5/models/best_model.pth
```


## Описание эксперимента
- Число эпох: 56 (прервано из-за нехватки ресурсов)
- Архитектура: resnet_model_v2 (src/model/)
- Dropout: 0.3
- scheduler: CosineAnnealingWarmRestarts

Поменял sheduler на косинусный, чтобы модель вышла из локального минимума

## Результаты

На тесте:

Test Loss: 1.1323 | accuracy: 0.6200 | precision: 0.6412 | recall: 0.6200 | f1: 0.6214


На валидации:

Val Loss: 1.0816 | accuracy: 0.6539 | precision: 0.6625 | recall: 0.6543 | f1: 0.6535

## Вывод

Из логов видно, что модель сперва прекратила учиться, а на обратном ходе sheduler-а снова начала обучаться. Это может говорить о том, что модель вышла из локального минимума.
Возможно, стоит изначально поставить большой lr, чтобы модель перескачила локальный минимум.