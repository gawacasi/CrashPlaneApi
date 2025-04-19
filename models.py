from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Prediction(db.Model):
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True)
    aboard = db.Column(db.Integer)
    ground = db.Column(db.Integer)
    year = db.Column(db.Integer)
    result = db.Column(db.Integer)  # 1 = acidente com fatalidades, 0 = sem fatalidades
