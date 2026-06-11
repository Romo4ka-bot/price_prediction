from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Optional
import os


from .models import User, Prediction, get_db
from .schemas import (
    UserCreate, UserLogin, UserResponse, Token,
    PredictionInput, PredictionResult, PredictionCreate, 
    PredictionResponse, PredictionUpdate, PredictionFilter,
    FeaturesResponse, FeatureConfig
)
from .auth import (
    authenticate_user, create_access_token, get_password_hash,
    get_current_active_user, ACCESS_TOKEN_EXPIRE_MINUTES
)
from .ml_model import get_predictor


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = FastAPI(
    title="Real Estate Price Predictor",
    description="API для предсказания стоимости недвижимости",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем статические файлы
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Инициализируем предиктор
predictor = get_predictor()


# ========== Главная страница ==========

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Главная страница"""
    with open(os.path.join(TEMPLATES_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()


# ========== Аутентификация ==========

@app.post("/api/auth/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    """Регистрация нового пользователя"""
    # Проверяем уникальность username
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Пользователь с таким именем уже существует"
        )
    
    # Проверяем уникальность email
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Пользователь с таким email уже существует"
        )
    
    # Создаем пользователя
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@app.post("/api/auth/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Вход в систему"""
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверное имя пользователя или пароль",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/api/auth/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_active_user)):
    """Получение информации о текущем пользователе"""
    return current_user


# ========== Предсказания ==========

@app.get("/api/features", response_model=FeaturesResponse)
def get_features():
    """Получение списка признаков для формы"""
    features = predictor.get_feature_list()
    return FeaturesResponse(features=features)


@app.post("/api/predict", response_model=PredictionResult)
def predict_price(
    data: PredictionInput,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Предсказание цены и сохранение в историю"""
    # Делаем предсказание
    # input_dict = data.dict()
    # result = predictor.predict(input_dict)
    result = predictor.predict(data)  # .to_dataframe())
    
    # Сохраняем в базу данных
    prediction = Prediction(
        user_id=current_user.id,
        input_data=data.model_dump(),  # input_dict,
        predicted_price=result['predicted_price'],
        lower_bound=result['lower_bound'],
        upper_bound=result['upper_bound'],
        confidence=result['confidence'],
        error_margin=result['error_margin'],
        district=data.district,  # input_dict.get('district'),
        area=data.area,  # input_dict.get('area'),
        rooms=data.rooms,  # input_dict.get('rooms')
    )
    db.add(prediction)
    db.commit()
    
    return PredictionResult(**result)


@app.get("/api/predictions", response_model=List[PredictionResponse])
def get_predictions(
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    district: Optional[str] = None,
    rooms: Optional[int] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Получение истории предсказаний с фильтрацией"""
    query = db.query(Prediction).filter(Prediction.user_id == current_user.id)
    
    # Применяем фильтры
    if date_from:
        query = query.filter(Prediction.created_at >= date_from)
    if date_to:
        query = query.filter(Prediction.created_at <= date_to)
    if district:
        query = query.filter(Prediction.district == district)
    if rooms:
        query = query.filter(Prediction.rooms == rooms)
    if min_price:
        query = query.filter(Prediction.predicted_price >= min_price)
    if max_price:
        query = query.filter(Prediction.predicted_price <= max_price)
    
    predictions = query.order_by(Prediction.created_at.desc()).offset(skip).limit(limit).all()
    return predictions


@app.get("/api/predictions/{prediction_id}", response_model=PredictionResponse)
def get_prediction(
    prediction_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Получение конкретного предсказания"""
    prediction = db.query(Prediction).filter(
        Prediction.id == prediction_id,
        Prediction.user_id == current_user.id
    ).first()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Предсказание не найдено")
    
    return prediction


@app.put("/api/predictions/{prediction_id}", response_model=PredictionResponse)
def update_prediction(
    prediction_id: int,
    update_data: PredictionUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Обновление данных предсказания (пересчет цены)"""
    prediction = db.query(Prediction).filter(
        Prediction.id == prediction_id,
        Prediction.user_id == current_user.id
    ).first()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Предсказание не найдено")
    
    # Если обновляем входные данные - пересчитываем цену
    if update_data.input_data:
        int_fields = {'floor', 'total_floors', 'passenger_elevators', 'cargo_elevators',
                      'rooms', 'separate_bathrooms'}
        float_fields = {'area', 'living_area', 'kitchen_area', 'ceiling_height'}
        cleaned = {}
        for k, v in update_data.input_data.items():
            if k in int_fields:
                cleaned[k] = int(v) if v != '' and v is not None else 0
            elif k in float_fields:
                cleaned[k] = float(v) if v != '' and v is not None else 0.0
            else:
                cleaned[k] = v
        new_result = predictor.predict(PredictionInput(**cleaned))
        prediction.input_data = update_data.input_data
        prediction.predicted_price = new_result['predicted_price']
        prediction.lower_bound = new_result['lower_bound']
        prediction.upper_bound = new_result['upper_bound']
        prediction.confidence = new_result['confidence']
        prediction.error_margin = new_result['error_margin']
        prediction.district = update_data.input_data.get('district')
        prediction.area = update_data.input_data.get('area')
        prediction.rooms = update_data.input_data.get('rooms')
        
        db.commit()
        db.refresh(prediction)
    
    return prediction


@app.delete("/api/predictions/{prediction_id}")
def delete_prediction(
    prediction_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Удаление предсказания"""
    prediction = db.query(Prediction).filter(
        Prediction.id == prediction_id,
        Prediction.user_id == current_user.id
    ).first()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Предсказание не найдено")
    
    db.delete(prediction)
    db.commit()
    
    return {"message": "Предсказание удалено"}


@app.get("/api/statistics")
def get_statistics(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Получение статистики по предсказаниям пользователя"""
    predictions = db.query(Prediction).filter(Prediction.user_id == current_user.id).all()
    
    if not predictions:
        return {
            "total_predictions": 0,
            "avg_price": 0,
            "min_price": 0,
            "max_price": 0,
            "by_district": {},
            "by_rooms": {}
        }
    
    prices = [p.predicted_price for p in predictions]
    
    # Статистика по районам
    by_district = {}
    for p in predictions:
        district = p.district or "Не указан"
        if district not in by_district:
            by_district[district] = {"count": 0, "avg_price": 0, "total": 0}
        by_district[district]["count"] += 1
        by_district[district]["total"] += p.predicted_price
    
    for district in by_district:
        by_district[district]["avg_price"] = int(
            by_district[district]["total"] / by_district[district]["count"]
        )
        del by_district[district]["total"]
    
    # Статистика по комнатам
    by_rooms = {}
    for p in predictions:
        rooms = p.rooms or 0
        if rooms not in by_rooms:
            by_rooms[rooms] = {"count": 0, "avg_price": 0, "total": 0}
        by_rooms[rooms]["count"] += 1
        by_rooms[rooms]["total"] += p.predicted_price
    
    for rooms in by_rooms:
        by_rooms[rooms]["avg_price"] = int(
            by_rooms[rooms]["total"] / by_rooms[rooms]["count"]
        )
        del by_rooms[rooms]["total"]
    
    return {
        "total_predictions": len(predictions),
        "avg_price": int(sum(prices) / len(prices)),
        "min_price": min(prices),
        "max_price": max(prices),
        "by_district": by_district,
        "by_rooms": by_rooms
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
