# репос для csi

## Описания скриптов

Фикс кривых csv
```
python tools/fix_csi_logs.py wifi_data_set --output-dir wifi_data_set_fixed
```

Проверка csv(путь до датасета исправляйте внутри файла)

```
python tools/validate_fixed_csi_dataset.py
```

Пример применения парсера
```
from tools.csi_parser import Parser
p = Parser('wifi_data_set_fixed/id_person_01/label_00/test_01/test1__dev1_64_E8_33_57_AA_F4.data').parse()
print(p.iloc[0])
```


------------------------
## Пайплайн предсказания бинарных меток на наличие/отсутствие движения
```
pipelines/binary_predictor.py
```

Пример в 
```
experiments_notebooks/binary_predictor.ipynb
```

Веса для этого в 
```
artifacts/classic_ml_majority_vote_metrics_each_esp
```