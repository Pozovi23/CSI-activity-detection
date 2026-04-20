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