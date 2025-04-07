from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

class Upload(db.Model):
    """Model for storing leaf analysis results"""
    __tablename__ = 'uploads'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    leaf_type = db.Column(db.String(50), default='Hibiscus')
    notes = db.Column(db.Text, nullable=True)
    
    def __init__(self, filename, prediction, confidence, leaf_type='Hibiscus', notes=None):
        self.filename = filename
        self.timestamp = datetime.utcnow()
        self.prediction = prediction
        self.confidence = confidence
        self.leaf_type = leaf_type
        self.notes = notes
    
    def __repr__(self):
        return f'<Upload {self.filename}: {self.prediction} ({self.confidence:.2f}%)>'
    
    def to_dict(self):
        """Convert the model instance to a dictionary"""
        return {
            'id': self.id,
            'filename': self.filename,
            'timestamp': self.timestamp.isoformat(),
            'prediction': self.prediction,
            'confidence': self.confidence,
            'leaf_type': self.leaf_type,
            'notes': self.notes
        }
