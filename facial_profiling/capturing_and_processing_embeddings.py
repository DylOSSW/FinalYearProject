import cv2
import sqlite3
from mtcnn import MTCNN
from datetime import datetime
import time
import threading
import queue
import speech_recognition as sr
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import warnings
import sys
from datetime import datetime, timedelta
import tensorflow as tf
import os

# Global queue for frames
frame_queue = queue.Queue()
# Event to signal the processing thread to terminate
stop_event = threading.Event()

def create_connection(db_file):
    """Create a connection to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(db_file, check_same_thread=False)
        print("SQLite version:", sqlite3.version)
    except Exception as e:
        print(e)
    return conn

def create_tables(conn):
    """Create tables as per the new schema."""
    user_profiles_table_sql = '''
    CREATE TABLE IF NOT EXISTS user_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
    );'''

    facial_embeddings_table_sql = '''
    CREATE TABLE IF NOT EXISTS facial_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        embedding BLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
        FOREIGN KEY (user_id) REFERENCES user_profiles (id)
    );'''

    facial_landmarks_table_sql = '''
    CREATE TABLE IF NOT EXISTS facial_landmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        left_eye TEXT NOT NULL,
        right_eye TEXT NOT NULL,
        nose TEXT NOT NULL,
        mouth_left TEXT NOT NULL,
        mouth_right TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
        FOREIGN KEY (user_id) REFERENCES user_profiles (id)
    );'''

    c = conn.cursor()
    c.execute(user_profiles_table_sql)
    c.execute(facial_embeddings_table_sql)
    c.execute(facial_landmarks_table_sql)
    conn.commit()


def insert_user_profile(conn, name="Anonymous"):
    """Insert a dummy/placeholder user profile into the database."""
    sql = 'INSERT INTO user_profiles (name) VALUES (?)'
    cur = conn.cursor()
    cur.execute(sql, (name,))
    conn.commit()
    return cur.lastrowid

def insert_embedding(conn, user_id, embedding):
    """Insert facial embedding into the database, associated with a user."""
    # Convert the embedding tensor to a NumPy array and then to bytes
    embedding_array = embedding.detach().numpy()
    embedding_bytes = embedding_array.tobytes()  
    sql = "INSERT INTO facial_embeddings (user_id, embedding) VALUES (?, ?)"
    cur = conn.cursor()
    cur.execute(sql, (user_id, embedding_bytes))
    conn.commit()

def insert_landmarks(conn, user_id, landmarks):
    """Insert facial landmarks into the database, associated with a user."""
    sql = 'INSERT INTO facial_landmarks (user_id, left_eye, right_eye, nose, mouth_left, mouth_right) VALUES (?, ?, ?, ?, ?, ?)'
    cur = conn.cursor()
    cur.execute(sql, (user_id, landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4]))
    conn.commit()

   
def capture_embeddings(detector, facenet_model, image):
    """Detect faces and capture facial embeddings using MTCNN and FaceNet."""
    try:
        results = detector.detect_faces(image)
        embeddings = []
        if results:
            for result in results:
                keypoints = result['keypoints']
                # Crop the face from the image
                cropped_face = image[result['box'][1]:result['box'][1]+result['box'][3],
                                     result['box'][0]:result['box'][0]+result['box'][2]]
                # Convert the cropped face to a PIL Image
                face_image = Image.fromarray(cropped_face)
                # Resize the face image to the required input size of the FaceNet model
                face_image = face_image.resize((160, 160))
                # Convert the resized face image to a PyTorch tensor
                face_tensor = transforms.ToTensor()(face_image).unsqueeze(0)
                # Generate the embedding using FaceNet model
                embedding = facenet_model(face_tensor)
                embeddings.append(embedding)
                print("Embeddings",results)
            return embeddings

        else:
            print("No faces detected.")
            return None
    except Exception as e:
        print(f"Error capturing embeddings: {e}")
        return None
    
def capture_landmarks(detector, image):
    """Detect and capture facial landmarks using MTCNN."""
    try:
        results = detector.detect_faces(image)
        print("landmarks",results)
        if results:
            keypoints = results[0]['keypoints']
            landmarks = (
                str(keypoints['left_eye']),
                str(keypoints['right_eye']),
                str(keypoints['nose']),
                str(keypoints['mouth_left']),
                str(keypoints['mouth_right'])
            )
            return landmarks
        else:
            print("No faces detected.")
            return None
    except Exception as e:
        print(f"Error capturing landmarks: {e}")
        return None

def process_frames(detector, facenet_model, conn):   
    user_name = input("Please enter the user's name: ")
    user_id = insert_user_profile(conn, user_name)
    while not stop_event.is_set() or not frame_queue.empty():
        if not frame_queue.empty():
            image = frame_queue.get()
            embeddings = capture_embeddings(detector, facenet_model, image)
            if embeddings:
                for embedding in embeddings:
                    insert_embedding(conn, user_id, embedding)
            landmarks = capture_landmarks(detector, image)
            if landmarks:
                insert_landmarks(conn, user_id, landmarks)
            frame_queue.task_done()


def main():
    db_file = 'Us.db'
    conn = create_connection(db_file)
    if conn is not None:
        create_tables(conn)
        cap = cv2.VideoCapture(0)
        detector = MTCNN()
        facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

        # Initialize last_snapshot_time here
        last_snapshot_time = time.time()

        # Start the processing thread
        processing_thread = threading.Thread(target=process_frames, args=(detector, facenet_model, conn))
        processing_thread.start()

        try:
            while True:
                ret, frame = cap.read()
                if ret:
                    cv2.imshow('Face Detection', frame)

                    current_time = time.time()
                    if current_time - last_snapshot_time >= 5:
                        last_snapshot_time = current_time
                        # Put only the current frame into the queue for processing
                        frame_queue.put(frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            stop_event.set()
            processing_thread.join()
            conn.close()
            cap.release()
            cv2.destroyAllWindows()
    else:
        print("Failed to create a database connection.")

if __name__ == '__main__':
    main()


