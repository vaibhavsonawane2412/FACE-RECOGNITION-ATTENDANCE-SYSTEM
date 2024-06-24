from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import base64
import sqlite3
import cv2
from datetime import datetime
import pytz
from face_extraction import recognize_face, train_model
import subprocess
from io import BytesIO
import csv

from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = 'your_secret_key'


# Database connection for users
def get_db_connection():
    db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database')
    db_path = os.path.join(db_dir, 'users.db')
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # This allows us to access rows as dictionaries
    return conn

# Function to retrieve users from the database
def get_users():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users')
        users = cursor.fetchall()
        conn.close()
        return users
    except Exception as e:
        print(f"Error fetching users: {e}")
        return []

@app.route('/user_list')
def user_list():
    if not is_logged_in():
        flash('Please log in to access this page.', 'error')
        return redirect(url_for('login'))
    
    # Fetch users from the database
    users = get_users()

    # Pass the users data to the template for rendering
    return render_template('admin/user_list.html', users=users)


# Check if user has admin privileges
def is_admin(user_id):
    return user_id == 1  # For demonstration purposes

# Check if user is logged in
def is_logged_in():
    return 'user_id' in session

# Admin login
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'adminpassword':
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('admin/admin_login.html')

@app.route('/admin')
def admin_dashboard():
    if not is_logged_in():
        flash('Please log in to access the admin dashboard.', 'error')
        return redirect(url_for('login')) 

    return render_template('admin/admin_dashboard.html')


@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    if not is_logged_in():
        flash('Please log in to access the admin dashboard.', 'error')
        return redirect(url_for('login'))  # Redirect to login page if not logged in

    if request.method == 'POST':
        # Extract form data
        name = request.form['name']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']

        with get_db_connection() as conn:
            user_exists = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
            if user_exists:
                flash('Username already exists. Please choose a different one.', 'error')
            else:
                conn.execute('INSERT INTO users (username, password, name, email) VALUES (?, ?, ?, ?)',
                             (username, password, name, email))
                flash('User added successfully!', 'success')
                return redirect(url_for('admin_dashboard'))  # Redirect to admin dashboard after successful submission

    return render_template('admin/add_user.html')


@app.route('/view_attendance')
def view_attendance():
    if not is_logged_in():
        flash('Please log in to view attendance records.', 'error')
        return redirect(url_for('login'))

    # Fetch attendance data from the database
    with get_db_connection() as conn:
        attendance_data = conn.execute('SELECT * FROM attendance').fetchall()

    # Pass the attendance data to the template for rendering
    return render_template('admin/view_attendance_records.html', attendance=attendance_data)

from flask import send_file, Response
from io import BytesIO
import pandas as pd
import os  # Added for temporary filename handling

@app.route('/export_attendance_excel')
def export_attendance_excel():
  # Fetch attendance data from the database
  try:
    conn = get_db_connection()  # Assuming you have a function to get the database connection
    cursor = conn.cursor()
    cursor.execute('SELECT user_id, timestamp FROM attendance')
    attendance_data = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]  # Fetch column names from cursor
  except Exception as e:
    return f"Error fetching attendance data: {str(e)}"

  # Create a DataFrame using pandas
  df = pd.DataFrame(attendance_data, columns=column_names)

  # Create a file-like object in memory
  excel_buffer = BytesIO()

  # Write the DataFrame to the Excel buffer
  df.to_excel(excel_buffer, index=False)

  # Set the file pointer to the beginning of the buffer
  excel_buffer.seek(0)

  # Option 1: Using temporary file (recommended)
  temp_filename = 'attendance_data.xlsx'
  with open(temp_filename, 'wb') as f:
      f.write(excel_buffer.getvalue())
  return send_file(temp_filename, as_attachment=True, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.route('/capture_samples')
def capture_samples_page():
    if not is_logged_in():
        flash('Please log in to capture samples.', 'error')
        return redirect(url_for('login'))
    return render_template('admin/capture_samples.html')
@app.route('/train_model_route')
def train_model_route():
    result = train_model(samples_dir)
    return jsonify({'status': 'success', 'message': result})

# Function to fetch attendance summary data from the database
def fetch_attendance_summary():
    conn = sqlite3.connect('your_database.db')  # Replace 'your_database.db' with your actual database path
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM attendance_summary')
    summary_data = cursor.fetchone()
    conn.close()
    return summary_data

@app.route('/attendance_summary')
def attendance_summary():
    summary_data = fetch_attendance_summary()
    return render_template('attendance_summary.html', summary_data=summary_data)


@app.route('/user_reports')
def user_reports():
    
    return render_template('admin/user_reports.html')

@app.route('/emotion_summary')
def emotion_summary():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, timestamp, emotion, COUNT(*) as count FROM emotion_detection GROUP BY emotion')
    emotion_data = cursor.fetchall()
    conn.close()
    return render_template('admin/emotion_summary.html', emotion_data=emotion_data)

@app.route('/view_emotion_detection')
def view_emotion_detection():
    conn = sqlite3.connect('database/users.db')
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM emotion_detection')
    results = cursor.fetchall()

    conn.close()

    return render_template('view_emotion_detection.html', results=results)


from flask import request, jsonify
import sqlite3

@app.route('/save_emotion', methods=['POST'])
def save_emotion():
    if not request.is_json:
        return jsonify({'success': False, 'error': 'Request must be JSON'}), 400
    
    data = request.json
    user_id = data.get('user_id')
    emotion = data.get('emotion')
    
    if user_id is None or emotion is None:
        return jsonify({'success': False, 'error': 'Missing user_id or emotion'}), 400
    
    # Perform additional validation or authorization checks here if needed
    
    conn = sqlite3.connect('database/users.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO emotion_detection (user_id, emotion) VALUES (?, ?)
        ''', (user_id, emotion))
        
        conn.commit()
        return jsonify({'success': True}), 200
    except sqlite3.Error as e:
        conn.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()


from flask import render_template
import sqlite3


@app.route('/age_gender_summary')
def age_gender_summary():
    try:
        conn = sqlite3.connect('database/users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, gender, age FROM age_gender_detection')
        all_age_gender_data = cursor.fetchall()
        conn.close()
        
        # Filter data to include only records with ids that are multiples of 1000
        filtered_age_gender_data = [
            record for record in all_age_gender_data if record[0] % 1000 == 0
        ]
        
        return render_template('admin/age_gender_summary.html', age_gender_data=filtered_age_gender_data)
    except sqlite3.Error as e:
        print("An error occurred while fetching data from the database:", e)
        return render_template('error.html', message="Failed to fetch data from the database.")


@app.route('/landmark_detection')
def landmark_detection():
    subprocess.Popen(['python', 'Landmark_Detection.py'])
    return render_template('admin/landmark.html')

@app.route('/landmark_summary')
def landmark_summary():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, timestamp, jawline_detected, eyebrows_detected, nose_detected, eyes_detected, lips_detected FROM landmarks_detection')
    all_landmark_data = cursor.fetchall()
    conn.close()
    
    # Filter data to include only every 100th entry (1, 100, 200, etc.)
    filtered_landmark_data = [
        record for idx, record in enumerate(all_landmark_data) if (idx + 1) % 100 == 1
    ]
    
    # Parse the filtered landmarks data
    parsed_landmark_data = []
    for record in filtered_landmark_data:
        record_id = record[0]
        timestamp = record[1]
        jawline_detected = bool(record[2])
        eyebrows_detected = bool(record[3])
        nose_detected = bool(record[4])
        eyes_detected = bool(record[5])
        lips_detected = bool(record[6])
        
        parsed_landmark_data.append({
            'id': record_id,
            'timestamp': timestamp,
            'jawline_detected': jawline_detected,
            'eyebrows_detected': eyebrows_detected,
            'nose_detected': nose_detected,
            'eyes_detected': eyes_detected,
            'lips_detected': lips_detected
        })
    
    return render_template('admin/landmark_summary.html', landmark_data=parsed_landmark_data)


@app.route('/training_status')
def training_status():
    # Your code here
    return render_template('training_status.html')

@app.route('/training_logs')
def training_logs():
    # Your code here
    return render_template('training_logs.html')
  

@app.route('/admin/attendance_records')
def admin_attendance_records():
    with get_db_connection() as conn:
        records = conn.execute('SELECT user_id, timestamp FROM attendance').fetchall()
    ist_tz = pytz.timezone('Asia/Kolkata')
    formatted_records = []
    for record in records:
        utc_time = datetime.strptime(record['timestamp'], '%Y-%m-%d %H:%M:%S.%f%z')
        ist_time = utc_time.astimezone(ist_tz)
        formatted_records.append({'user_id': record['user_id'], 'timestamp': ist_time.strftime('%Y-%m-%d %H:%M:%S')})
    return render_template('admin/admin_attendance_records.html', records=formatted_records)

@app.route('/admin/manage_users')
def admin_manage_users():
    with get_db_connection() as conn:
        users = conn.execute('SELECT * FROM users').fetchall()
    return render_template('admin/admin_manage_users.html', users=users)

@app.route('/admin/clear_attendance', methods=['POST'])
def admin_clear_attendance():
    if 'user_id' not in session or not is_admin(session['user_id']):
        flash('You are not authorized to access this page.', 'error')
        return redirect(url_for('login'))
    with get_db_connection() as conn:
        conn.execute('DELETE FROM attendance')
    flash('All attendance records have been cleared.', 'success')
    return redirect(url_for('admin_attendance_records'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        name = request.form['name']
        email = request.form['email']

        with get_db_connection() as conn:
            user_exists = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
            if user_exists:
                flash('Username already exists. Please choose a different one.', 'error')
            else:
                conn.execute('INSERT INTO users (username, password, name, email) VALUES (?, ?, ?, ?)',
                             (username, password, name, email))
                flash('Registration successful. You can now log in.', 'success')
                return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        with get_db_connection() as conn:
            user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        if user and password == user['password']:
            session['user_id'] = user['id']
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('admin_logged_in', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('admin_login'))

@app.route('/')
def index():
    if 'user_id' in session:
        return render_template('index.html')
    return redirect(url_for('login'))


@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

@app.route('/emotion')
def emotion():
    subprocess.Popen(['python', 'emotion_detection.py'])
    return render_template('admin/emotion.html')


@app.route('/emotion_stream')
def emotion_stream():
    # Run the emotion detection script and stream the output to the client
    process = subprocess.Popen(['python', 'emotion_detection.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def generate():
        while True:
            line = process.stdout.readline()
            if not line:
                break
            yield line.strip() + b'\n'

    return Response(generate(), mimetype='text/plain')

from flask import request, jsonify


samples_dir = os.path.join('face_attendance_system', 'samples')
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

@app.route('/save_sample', methods=['POST'])
def save_sample():
    data = request.get_json()
    img_data = data['image']
    sample_num = data['sample_num']
    username = data['username']
    img_data = base64.b64decode(img_data.split(',')[1])
    user_dir = os.path.join(samples_dir, username)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    sample_path = os.path.join(user_dir, f'sample_{sample_num}.jpg')
    with open(sample_path, 'wb') as f:
        f.write(img_data)
    return jsonify({'status': 'success', 'sample_num': sample_num})


# Route to mark attendance
@app.route('/mark_attendance', methods=['GET', 'POST'])
def mark_attendance():
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    camera.release()
    if not ret:
        return jsonify({'status': 'error', 'message': 'Could not capture image from webcam'})
    temp_image_path = 'temp_image.jpg'
    cv2.imwrite(temp_image_path, frame)
    username, result_image_path = recognize_face(temp_image_path)
    os.remove(temp_image_path)
    if username:
        with get_db_connection() as conn:
            current_utc_time = datetime.now(pytz.utc)
            ist_tz = pytz.timezone('Asia/Kolkata')
            ist_time = current_utc_time.astimezone(ist_tz)
            conn.execute('INSERT INTO attendance (user_id, timestamp) VALUES (?, ?)', (username, str(ist_time)))
        return jsonify({'status': 'success', 'message': f'Attendance marked successfully for user {username}'})
    else:
        return jsonify({'status': 'error', 'message': 'Face not recognized'})



@app.route('/attendance_records')
def attendance_records():
    with get_db_connection() as conn:
        records = conn.execute('SELECT user_id, timestamp FROM attendance').fetchall()
    ist_tz = pytz.timezone('Asia/Kolkata')
    formatted_records = []
    for record in records:
        utc_time = record['timestamp']
        ist_time = utc_time.astimezone(ist_tz)
        formatted_records.append({'user_id': record['user_id'], 'timestamp': ist_time.strftime('%Y-%m-%d %H:%M:%S')})
    return jsonify(formatted_records)

@app.route('/age_gender')
def age_gender():
    subprocess.Popen(['python', 'age_gender_detection.py'])
    results = []
    try:
        with open('age_gender_results.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                gender, age = line.strip().split(', ')
                gender = gender.split(': ')[1]
                age = age.split(': ')[1]
                results.append((gender, age))
    except FileNotFoundError:
        results = [("No data yet", "No data yet")]
    return render_template('admin/age_gender.html', results=results)




if __name__ == '__main__':
    app.run(debug=True)
