from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

from utils.project_paths import ROOT_DIR

DB_PATH = os.environ.get(
    'PRICE_PREDICTION_DB_PATH',
    os.path.join(ROOT_DIR, 'deploy', 'data', 'real_estate.db'),
)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Integer, default=1)
    
    # Отношение к предсказаниям
    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Входные данные
    input_data = Column(JSON, nullable=False)  # Сохраняем все признаки
    
    # Результаты предсказания
    predicted_price = Column(Integer, nullable=False)
    lower_bound = Column(Integer, nullable=False)
    upper_bound = Column(Integer, nullable=False)
    confidence = Column(String, nullable=False)
    error_margin = Column(String, nullable=False)
    
    # Дополнительные поля для фильтрации
    property_type = Column(String, default="Квартира")
    district = Column(String)
    area = Column(Float)
    rooms = Column(Integer)
    
    # Отношение к пользователю
    user = relationship("User", back_populates="predictions")


# Создаем таблицы
Base.metadata.create_all(bind=engine)


# Dependency для получения сессии БД
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
