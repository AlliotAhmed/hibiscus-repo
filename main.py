import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Initialize db connection first to avoid circular imports
from models import db

# Create a new Flask app instance and configure it
from flask import Flask
app = Flask(__name__)

# Configure the database connection
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Required app configuration 
app.secret_key = os.environ.get("SESSION_SECRET", "hibiscus-leaf-classifier-secret")

# Initialize the database with this app
db.init_app(app)

# Create database tables
with app.app_context():
    db.create_all()
    logging.info("Database tables created successfully")

# Import the routes from app.py - must be after database setup
import app as app_routes

# Get the Flask app with routes from app.py
app = app_routes.app

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
