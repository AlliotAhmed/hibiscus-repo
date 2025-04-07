import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import io
import base64

from ml_model import predict_disease
from utils import allowed_file, preprocess_image, create_upload_folder

# Import models
from models import Upload, db

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Configure upload folder
UPLOAD_FOLDER = create_upload_folder()

# This function will be called from main.py to avoid circular imports
def setup_routes(app):
    # Set up WSGI proxy fix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
    
    # Configure upload folder
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

    # Define routes
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/upload', methods=['POST'])
    def upload_file():
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                # Secure the filename
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Save the file
                file.save(file_path)
                
                # Preprocess the image for the model
                processed_image = preprocess_image(file_path)
                
                # Make prediction
                prediction, confidence = predict_disease(processed_image)
                
                # Save result to database
                try:
                    upload_entry = Upload(
                        filename=filename,
                        prediction=prediction['label'],
                        confidence=prediction['confidence'],
                        leaf_type='Hibiscus',
                        notes=prediction['info']
                    )
                    db.session.add(upload_entry)
                    db.session.commit()
                    logging.info(f"Saved analysis result to database: {upload_entry}")
                except Exception as db_error:
                    logging.error(f"Failed to save to database: {str(db_error)}")
                    db.session.rollback()
                
                # Create a base64 representation of the image for display
                with open(file_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                
                return render_template('index.html', 
                                    prediction=prediction, 
                                    confidence=confidence,
                                    image_data=img_data,
                                    filename=filename)
                                    
            except Exception as e:
                logging.error(f"Error processing file: {str(e)}")
                flash(f'Error processing file: {str(e)}', 'danger')
                return redirect(url_for('index'))
        else:
            flash('File type not allowed. Please upload a JPG or PNG image.', 'warning')
            return redirect(url_for('index'))

    @app.route('/history')
    def history():
        """View analysis history from database"""
        uploads = Upload.query.order_by(Upload.timestamp.desc()).all()
        return render_template('history.html', uploads=uploads)

    @app.route('/api/history')
    def api_history():
        """API endpoint to get analysis history as JSON"""
        uploads = Upload.query.order_by(Upload.timestamp.desc()).all()
        return jsonify([upload.to_dict() for upload in uploads])

    @app.errorhandler(413)
    def too_large(e):
        flash('File too large. Maximum size is 16MB.', 'danger')
        return redirect(url_for('index'))

    @app.errorhandler(500)
    def server_error(e):
        flash('Server error occurred. Please try again.', 'danger')
        return redirect(url_for('index'))
        
    return app

# This section will only run if app.py is executed directly
if __name__ == '__main__':
    # Create a Flask application for direct execution
    flask_app = Flask(__name__)
    flask_app.secret_key = os.environ.get("SESSION_SECRET", "hibiscus-leaf-classifier-secret")
    
    # Configure the database
    flask_app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
    flask_app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_recycle": 300,
        "pool_pre_ping": True,
    }
    flask_app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    
    # Initialize the database
    db.init_app(flask_app)
    
    # Set up routes
    app = setup_routes(flask_app)
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)
