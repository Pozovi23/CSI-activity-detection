# Пайплайн рилтайма
-------------------
```
pip install pyserial
```

Выводите порты, к которым подключены есп:
```
python3 test.py
```

Используя эти порты, выполняем:
```
python3 data_collect/receiver.py --output ./csi_data --window 100 --overlap 50 --ports /dev/ttyUSB0 /dev/ttyUSB1 /dev/ttyUSB2 --baud 115200
```

window - колво измерений в файле  
overlap - колво пересечений таймстемпов в двух соседних файлах

-------------------
Запуск пайплайна детекции:

watch-dir - директория куда срутся файлы  
poll-interval - время задержки  
artifacts-dir - путь до весов  
```
python3 pipelines/binary_stream_predictor.py --watch-dir ./csi_data --poll-interval 1.0 --artifacts-dir artifacts/one_person_classic_ml_majority_vote_metrics_each_esp
```

