from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import bcrypt
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=True)
    password = Column(String(255), nullable=False)
    is_admin = Column(Integer, default=0)  # 0: 普通ユーザー, 1: 管理者

class Feedback(Base):
    __tablename__ = 'feedbacks'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    feedback = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class ImageClassification(Base):
    __tablename__ = 'image_classifications'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    image_path = Column(String(255), nullable=False)
    classification_result = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class ActivityLog(Base):
    __tablename__ = 'activity_logs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    activity = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class UserSettings(Base):
    __tablename__ = 'user_settings'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, unique=True)
    notify_tts = Column(Integer, default=1)  # 1: 有効, 0: 無効
    notify_classification = Column(Integer, default=1)
    notify_feedback = Column(Integer, default=1)

# データベース接続設定
DATABASE_URL = "sqlite:///ultimate_fusion_app.db"  # 適宜変更

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# テーブルの作成
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ユーザー関連操作
def add_user(username, name, password, email, db):
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    user = User(username=username, name=name, email=email, password=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)

def user_exists(username, db):
    return db.query(User).filter(User.username == username).first() is not None

def email_exists(email, db):
    return db.query(User).filter(User.email == email).first() is not None

def add_feedback(user_id, feedback, db):
    fb = Feedback(user_id=user_id, feedback=feedback)
    db.add(fb)
    db.commit()
    db.refresh(fb)

def get_all_feedback(db):
    return db.query(Feedback).order_by(Feedback.timestamp.desc()).all()

def add_activity_log(user_id, activity, db):
    log = ActivityLog(user_id=user_id, activity=activity)
    db.add(log)
    db.commit()
    db.refresh(log)

def get_user_activities(user_id, db):
    return db.query(ActivityLog).filter(ActivityLog.user_id == user_id).order_by(ActivityLog.timestamp.desc()).all()

def is_admin_user(username, db):
    user = db.query(User).filter(User.username == username).first()
    return user.is_admin == 1 if user else False

def add_image_classification(user_id, image_path, classification_result, db):
    cls = ImageClassification(user_id=user_id, image_path=image_path, classification_result=classification_result)
    db.add(cls)
    db.commit()
    db.refresh(cls)

def get_all_image_classifications(db):
    return db.query(ImageClassification).order_by(ImageClassification.timestamp.desc()).all()

def get_user_settings(user_id, db):
    return db.query(UserSettings).filter(UserSettings.user_id == user_id).first()

def update_user_settings(user_id, notify_tts, notify_classification, notify_feedback, db):
    settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
    if settings:
        settings.notify_tts = notify_tts
        settings.notify_classification = notify_classification
        settings.notify_feedback = notify_feedback
    else:
        settings = UserSettings(user_id=user_id, notify_tts=notify_tts, notify_classification=notify_classification, notify_feedback=notify_feedback)
        db.add(settings)
    db.commit()
    db.refresh(settings)
