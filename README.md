# Face Recognition Attendance System

Welcome to the Face Recognition Attendance System repository! This project utilizes computer vision techniques for automated attendance marking and real-time emotion detection.

## Overview

The Face Recognition Attendance System is designed to streamline the process of marking attendance using facial recognition technology. It also provides insights into real-time emotion detection and can estimate age and gender based on facial features.

## Features

- **Face Recognition:** Automates attendance marking through facial recognition.
- **Real-time Emotion Detection:** Detects emotions in real-time using machine learning models.
- **Age and Gender Estimation:** Estimates age and gender based on facial analysis.
- **Facial Landmark Detection:** Identifies and marks facial landmarks for detailed analysis.
- **Admin Dashboard:** Provides summaries and insights into attendance, emotions, age, and gender data.

## Getting Started

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vaibhavsonawane2412/FACE-RECOGNITION-ATTENDANCE-SYSTEM.git
   cd FACE-RECOGNITION-ATTENDANCE-SYSTEM```
   
2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate```

3. **Install dependencies:**
  ```bash
  pip install -r requirements.txt ```

### Database Initialization
  Initialize the SQLite database:
  ```bash
  python init_db.py```

### Running the Application
Start the Flask application:
```bash
python app.py```

### Access the application:
Open your web browser and go to
```bash
http://127.0.0.1:5000```