import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


# ========== Аутентификация ==========

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    password: str = Field(..., min_length=6)


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    username: Optional[str] = None


# ========== Предсказания ==========

# class PredictionInput(BaseModel):
#     area: float = Field(..., gt=0, description="Площадь в м²")
#     rooms: int = Field(..., ge=1, le=6, description="Количество комнат")
#     floor: int = Field(..., ge=1, description="Этаж")
#     total_floors: int = Field(..., ge=1, description="Всего этажей")
#     district: str = Field(..., description="Район")
#     year_built: int = Field(..., ge=1950, le=2024, description="Год постройки")
#     house_type: str = Field(..., description="Тип дома")
#     repair: str = Field(..., description="Ремонт")
#     distance_to_metro: float = Field(..., ge=0, description="Расстояние до метро в м")


class PredictionInput(BaseModel):
    # Числовые признаки (num_cols из конфига)
    area: float = Field(..., gt=0, description="Общая площадь в м²")
    living_area: float = Field(..., ge=0, description="Жилая площадь в м²")
    kitchen_area: float = Field(..., ge=0, description="Площадь кухни в м²")
    floor: int = Field(..., ge=1, description="Этаж")
    total_floors: int = Field(..., ge=1, description="Этажей в доме")
    passenger_elevators: int = Field(..., ge=0, description="Лифт пассажирский (кол-во)")
    cargo_elevators: int = Field(..., ge=0, description="Лифт грузовой (кол-во)")
    rooms: int = Field(..., ge=1, description="Количество комнат")
    ceiling_height: float = Field(..., gt=0, description="Высота потолков в м")
    separate_bathrooms: int = Field(..., ge=0, description="Кол-во раздельных санузлов")

    # Категориальные признаки (cat_cols из конфига)
    sale_type: str = Field(..., description="Тип продажи")
    object_type: str = Field(..., description="Объект продажи")
    garbage_chute: str = Field(..., description="Мусоропровод")
    parking: str = Field(..., description="Парковка")
    house_type: str = Field(..., description="Тип дома")
    window_view: str = Field(..., description="Вид из окон")
    distance_to_metro: str = Field(..., description="Расстояние до метро")
    district: str = Field(..., description="Округ")
    neighborhood: str = Field(..., description="Район")

    def to_dataframe(self) -> pd.DataFrame:
        """Конвертирует входные данные в DataFrame для DataTransformer"""
        data = {
            # Числовые колонки (порядок важен!)
            'Стоимость': [0.0],  # placeholder, будет заменен при предсказании
            'Общая площадь': [self.area],
            'Жилая площадь': [self.living_area],
            'Площадь кухни': [self.kitchen_area],
            'Этаж': [self.floor],
            'Этажей в доме': [self.total_floors],
            'Лифт пассажирский (кол-во)': [self.passenger_elevators],
            'Лифт грузовой (кол-во)': [self.cargo_elevators],
            'Количество комнат': [self.rooms],
            'Высота потолков': [self.ceiling_height],
            'Кол-во раздельных санузлов': [self.separate_bathrooms],

            # Категориальные колонки
            'Тип продажи': [self.sale_type],
            'Объект продажи': [self.object_type],
            'Мусоропровод': [self.garbage_chute],
            'Парковка': [self.parking],
            'Тип дома': [self.house_type],
            'Вид из окон': [self.window_view],
            'Расстояние до метро': [self.distance_to_metro],
            'Округ': [self.district],
            'Район': [self.neighborhood],
        }
        return pd.DataFrame(data)


class PredictionResult(BaseModel):
    predicted_price: int
    lower_bound: int
    upper_bound: int
    confidence: str
    error_margin: str


class PredictionCreate(BaseModel):
    input_data: Dict[str, Any]
    result: PredictionResult


class PredictionResponse(BaseModel):
    id: int
    user_id: int
    created_at: datetime
    input_data: Dict[str, Any]
    predicted_price: int
    lower_bound: int
    upper_bound: int
    confidence: str
    error_margin: str
    district: Optional[str]
    area: Optional[float]
    rooms: Optional[int]
    
    class Config:
        from_attributes = True


class PredictionUpdate(BaseModel):
    input_data: Optional[Dict[str, Any]] = None


# ========== Фильтрация и списки ==========

class PredictionFilter(BaseModel):
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    district: Optional[str] = None
    rooms: Optional[int] = None
    min_price: Optional[int] = None
    max_price: Optional[int] = None


class PredictionList(BaseModel):
    items: List[PredictionResponse]
    total: int


# ========== Конфигурация признаков ==========

class FeatureConfig(BaseModel):
    key: str
    name: str
    type: str
    required: bool
    min: Optional[float] = None
    max: Optional[float] = None
    unit: Optional[str] = None
    options: Optional[List[Any]] = None


class FeaturesResponse(BaseModel):
    features: List[FeatureConfig]
