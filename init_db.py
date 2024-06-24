import os
import sqlite3

def init_db():
    # Ensure the database directory exists
    db_dir = 'database'
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    # Connect to the SQLite database (or create it if it doesn't exist)
    db_path = os.path.join(db_dir, 'users.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Create the users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password TEXT,
                name TEXT,
                email TEXT
            )
        ''')

        # Create the attendance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'Present',
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        # Create the emotion detection table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotion_detection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                emotion TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        # Create the age_gender_detection table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS age_gender_detection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gender TEXT,
                age TEXT
            )
        ''')

        # Commit the changes
        conn.commit()
    except sqlite3.Error as e:
        print(f"An error occurred while initializing the database: {e}")
    finally:
        # Close the connection
        conn.close()

if __name__ == '__main__':
    init_db()
