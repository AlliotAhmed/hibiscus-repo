import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Import and configure Flask application
from flask import Flask

# Import the models and db
from models import db, Upload

# Initialize Flask app
app = Flask(__name__)

# Required app configuration 
app.secret_key = os.environ.get("SESSION_SECRET", "hibiscus-leaf-classifier-secret")

# Configure the database connection
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///hibiscus.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize the database with this app
db.init_app(app)

# Create database tables
try:
    with app.app_context():
        db.create_all()
        logging.info("Database tables created successfully")
except Exception as e:
    logging.error(f"Failed to create database tables: {str(e)}")

# Import the routes after database setup (using import_views function to avoid circular imports)
def import_views():
    from app import setup_routes
    setup_routes(app)

# Call the import_views function to register routes
import_views()

# For Vercel deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
