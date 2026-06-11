# Предсказание стоимости квартир

Система оценки стоимости недвижимости на основе трансформерной модели TabM. Включает веб-интерфейс с авторизацией и историей предсказаний.

## Установка

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Запуск

```bash
python deploy/run.py
```

Приложение доступно по адресу `http://localhost:8000`.

## Обучение

```bash
# TabM (основная модель)
python executors/tabm_trainer.py

# CatBoost (бейзлайн)
python executors/cb_trainer.py
```

## Конфигурация

| Переменная | По умолчанию | Описание |
|---|---|---|
| `PRICE_PREDICTION_MODEL_DIR` | `веса и конфиги/transformer/` | Каталог с весами модели |
| `PRICE_PREDICTION_DB_PATH` | `deploy/data/real_estate.db` | Путь к SQLite базе |

Каталог модели должен содержать `TablePredictor.pt` и `logs/config.json`.
