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
from cryptography.fernet import Fernet
#from config_key import key 
import tensorflow as tf

#cipher_suite = Fernet(key)

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
    """Create database tables."""
    session_table_sql = '''
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL
    );
    '''

    facial_embeddings_table_sql = '''
    CREATE TABLE IF NOT EXISTS facial_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        embedding BLOB NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions (id)
    );
    '''

    c = conn.cursor()
    c.execute(session_table_sql)
    c.execute(facial_embeddings_table_sql)

def insert_session(conn, timestamp):
    """Insert a new session into the database."""
    try:
        sql = 'INSERT INTO sessions (timestamp) VALUES (?)'
        cur = conn.cursor()
        cur.execute(sql, (timestamp,))
        conn.commit()
        return cur.lastrowid
    except Exception as e:
        print(f"Error inserting session: {e}")

def insert_embedding(conn, session_id, embedding):
    """Insert facial embedding into the database."""
    try:
        # Convert the embedding tensor to a NumPy array and then to bytes
        embedding_array = embedding.detach().numpy()
        embedding_bytes = embedding_array.tobytes()


        
        sql = "INSERT INTO facial_embeddings (session_id, embedding) VALUES (?, ?)"
        cur = conn.cursor()
        cur.execute(sql, (session_id, embedding_bytes))
        conn.commit()
    except Exception as e:
        print(f"Error inserting embedding: {e}")
   
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
            return embeddings
        else:
            print("No faces detected.")
            return None
    except Exception as e:
        print(f"Error capturing embeddings: {e}")
        return None

def process_frames(detector, facenet_model, conn):
    """Thread function to process frames and insert session IDs only when embeddings are captured."""
    while not stop_event.is_set() or not frame_queue.empty():
        if not frame_queue.empty():
            image = frame_queue.get()  # Adjusted to only get the image
            embeddings = capture_embeddings(detector, facenet_model, image)
            if embeddings:
                # Generate a new session ID only if embeddings are captured
                session_id = insert_session(conn, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                for embedding in embeddings:
                    insert_embedding(conn, session_id, embedding)
            frame_queue.task_done()

def main():
    db_file = 'mtcnn3.db'
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
