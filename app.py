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

# Import models - moved lower to avoid circular imports
from models import Upload

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Create the Flask app - This will be imported by main.py
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "hibiscus-leaf-classifier-secret")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Note: The database configuration is now done in main.py

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = create_upload_folder()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

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
            
            # Import the db from main to avoid circular imports
            from main import db

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
    # Import the db from main to avoid circular imports
    from main import db
    uploads = Upload.query.order_by(Upload.timestamp.desc()).all()
    return render_template('history.html', uploads=uploads)

@app.route('/api/history')
def api_history():
    """API endpoint to get analysis history as JSON"""
    # Import the db from main to avoid circular imports
    from main import db
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
