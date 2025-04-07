# Hibiscus Leaf Disease Classifier

A web application that uses machine learning to classify hibiscus leaves as healthy or diseased from uploaded images.

## Features

- Upload and analyze hibiscus leaf images
- ML-powered detection of leaf health status
- Detailed classification results with confidence scores
- History tracking of all analyses
- Responsive, user-friendly interface

## Technology Stack

- **Backend**: Flask (Python)
- **Database**: PostgreSQL
- **ML Model**: TensorFlow Lite
- **Image Processing**: Pillow, NumPy
- **Styling**: Bootstrap CSS
- **Deployment**: Replit

## Usage

1. Upload an image of a hibiscus leaf through the web interface
2. The application will process the image and provide a classification (Healthy/Diseased)
3. View detailed results including confidence percentage
4. Access the history page to see all past analyses

## Development

The application uses:
- A TensorFlow Lite model for leaf health classification
- Flask for web framework and routing
- PostgreSQL database for storing analysis history
- Image preprocessing pipeline for optimal ML model input

## Deployment

This application is ready to be deployed on Replit with the following requirements:
- Python 3.11+
- PostgreSQL database
- All dependencies listed in pyproject.toml

## License

This project is for educational purposes only.
