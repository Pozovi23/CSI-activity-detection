# CSI WiFi Works

Набор экспериментов для CSI WiFi в корне проекта.

Рабочие entrypoint-скрипты:

- [run_binary_recording_split.py](/home/kirill/camp/run_binary_recording_split.py)
- [run_binary_person_kfold.py](/home/kirill/camp/run_binary_person_kfold.py)
- [run_distance_recording_split.py](/home/kirill/camp/run_distance_recording_split.py)
- [run_distance_person_kfold.py](/home/kirill/camp/run_distance_person_kfold.py)

Общий модуль:

- [csi_wifi_common.py](/home/kirill/camp/csi_wifi_common.py)

Старые файлы `train_*` можно считать историческими. Для воспроизводимых запусков использовать нужно именно `run_*`.

## Что лежит в проекте

В корне должны быть:

- `wifi_data_set_fixed/`
- `tools/`
- `requirements.txt`

Если `wifi_data_set_fixed/` ещё нет, его нужно собрать из сырого датасета:

```bash
python tools/fix_csi_logs.py wifi_data_set --output-dir wifi_data_set_fixed
```

## Шаг 1. Установить зависимости

```bash
python -m pip install -r requirements.txt
```

Минимально нужны:

- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `tqdm`
- `torch`

## Шаг 2. Проверить, что fixed-датасет готов

Ожидаемая структура:

- `wifi_data_set_fixed/id_person_XX/label_YY/test_ZZ/*.data`

Если нужно быстро проверить структуру:

```bash
python tools/validate_fixed_csi_dataset.py
```

## Шаг 3. Понять, какой эксперимент нужен

### Бинарная задача

Цель:

- отличить `no_motion` от `motion`

Есть два режима.

#### 3.1. Обычный split по recording

```bash
python run_binary_recording_split.py
```

Что это проверяет:

- умеет ли модель обобщаться на новые записи

Что важно в выводе:

- `val_bal_acc`
- `test_bal_acc`
- `classification_report`

#### 3.2. Leave-one-person-out

```bash
python run_binary_person_kfold.py
```

Что это проверяет:

- умеет ли модель обобщаться на новых людей

Что важно в выводе:

- `test_bal_acc` по каждому человеку
- итоговые `mean_test_bal_acc` и `std_test_bal_acc`

Если обычный split хороший, а `person_kfold` сильно хуже, значит модель держится на человек-специфичных паттернах.

### Классификация расстояния

Цель:

- различать расстояния `0m / 1m / 2m / 3m`

В distance-задаче сейчас оставлен только один подход:

- ортогональная многоклассовая классификация через `CrossEntropy`

Есть два режима.

#### 3.3. Обычный split по recording

```bash
python run_distance_recording_split.py
```

Что это проверяет:

- насколько модель различает дистанции на новых записях

Что важно в выводе:

- `accuracy`
- `balanced_accuracy`
- `mae`
- `classification_report`

#### 3.4. Leave-one-person-out

```bash
python run_distance_person_kfold.py
```

Что это проверяет:

- насколько различение дистанций переносится на новых людей

Что важно в выводе:

- `test_bal_acc`
- `test_mae`
- средние значения по fold'ам

## Шаг 4. Понять, что делает pipeline

Все 4 entrypoint-скрипта используют один и тот же pipeline из [csi_wifi_common.py](/home/kirill/camp/csi_wifi_common.py).

### 4.1. Индексация записей

Из `wifi_data_set_fixed/` собирается таблица записей.

Одна запись = один `test_*` со всеми тремя устройствами:

- `dev1`
- `dev2`
- `dev3`

### 4.2. Split без утечки

До любых data-dependent шагов данные делятся на:

- `train`
- `val`
- `test`

Обычные скрипты:

- split по `recording_id`

`person_kfold`:

- один человек полностью уходит в `test`

Утечка контролируется через пересечения:

- по `recording_id`
- по `person`

### 4.3. Парсинг CSI

Из каждого `.data` файла читаются IQ-значения и переводятся в amplitude:

`amplitude = sqrt(imag^2 + real^2)`

Дальше модель использует только amplitude.

### 4.4. Удаление `null_subcarriers`

На `train` ищутся поднесущие, которые почти всегда нулевые.

Они:

- вычисляются только на `train`
- потом удаляются из `train/val/test`

Это нужно, чтобы модель не тратила параметры на заведомо пустые каналы.

### 4.5. Сигнальный препроцесс

На запись `[T, D, S]` применяются:

- `Hampel` filter по времени
- `Savitzky-Golay` smoothing по времени

Что сейчас не используется:

- detrend
- robust z-score внутри записи

### 4.6. Нарезка на окна

Каждая запись режется на окна:

- `window = 20`
- `step = 20`

То есть окна не пересекаются.

Форма данных:

- до reshape: `[N, T, D, S]`
- после reshape: `[N, C, T]`, где `C = D * S`

### 4.7. Нормализация

Сначала на `train` считаются:

- `mean`
- `std`

Потом те же значения используются для:

- `train`
- `val`
- `test`

Это последняя часть препроцесса перед моделью.

## Шаг 5. Понять модель

Модель специально оставлена минимальной и читаемой:

1. `SincConv1d`
2. `mean(abs(.))` по времени
3. линейная голова

Бинарная задача:

- один выходной логит

Distance-задача:

- 4 logits по классам расстояния

Идея такая:

- `SincConv1d` учит временные полосовые фильтры
- дальше берётся средняя энергия отклика по времени
- линейная голова решает задачу классификации

## Шаг 6. Почему до первой эпохи бывает долго

До обучения самые дорогие шаги такие:

1. чтение `.data` файлов
2. парсинг amplitude
3. поиск `null_subcarriers`
4. сборка окон после препроцесса

Для этого уже добавлен `tqdm`:

- на поиск `null_subcarriers`
- на построение окон
- на эпохи
- на батчи

Если задержка большая до первой эпохи, почти всегда причина именно в парсинге и построении окон, а не в PyTorch.

## Шаг 7. Как читать метрики

### Бинарная задача

Главные поля:

- `val_bal_acc`
- `test_bal_acc`

Это `balanced accuracy`, то есть средний recall по классам.

Смысл:

- не даёт метрике выглядеть хорошо только за счёт частого класса

### Distance-задача

Главные поля:

- `accuracy`
- `balanced_accuracy`
- `mae`

Смысл:

- `accuracy`: ровно правильный класс
- `balanced_accuracy`: одинаковое внимание ко всем дистанциям
- `mae`: средняя ошибка по расстоянию в метрах

Для distance-задачи `mae` особенно полезен, потому что:

- ошибка `2m -> 3m` лучше, чем `2m -> 0m`

## Шаг 8. Как читать таблицу фильтров

Во всех скриптах печатается таблица обученных `sinc`-фильтров.

Основные поля:

- `low_hz`
- `high_hz`
- `center_hz`
- `bandwidth_hz`

И ещё средняя активация по классам:

- бинарная задача: `act_no_motion`, `act_motion`
- distance-задача: `act_0m`, `act_1m`, `act_2m`, `act_3m`

Что это показывает:

- какие частотные полосы модель реально использует
- какие полосы активнее реагируют на движение
- какие полосы активнее реагируют на конкретную дистанцию

## Шаг 9. Рекомендуемый порядок работы

1. Проверить наличие `wifi_data_set_fixed/`
2. Установить зависимости
3. Прогнать `run_binary_recording_split.py`
4. Прогнать `run_binary_person_kfold.py`
5. Если бинарная задача стабильна, переходить к distance:
6. Прогнать `run_distance_recording_split.py`
7. Прогнать `run_distance_person_kfold.py`
8. Сравнить:
   - обычный split против `person_kfold`
   - `balanced accuracy`
   - `mae`
   - таблицы `sinc`-фильтров
