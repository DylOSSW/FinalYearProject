import cv2
import sqlite3
#from mtcnn import MTCNN
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
import mediapipe as mp
import torch
import pyttsx3
from openai import OpenAI
from cryptography.fernet import Fernet


# Initialize speech recognition and text-to-speech engines
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


# Global queue for frames
frame_queue = queue.Queue()
# Event to signal the processing thread to terminate
stop_event = threading.Event()

def load_encryption_key():
    with open('config.key', 'rb') as key_file:
        key = key_file.read()
        print(key)
    # Create a Fernet object with the loaded key
    return Fernet(key)

# Use the Fernet object for encryption
encryption_tool = load_encryption_key()

def load_openai_key():
    with open('openai_api.key', 'rb') as key_file:
        key = key_file.read().decode('utf-8')  # Decode bytes to string
    return key

api_key = load_openai_key()

# Initialize the OpenAI client with the api_key as a named (keyword) argument
client = OpenAI(api_key=api_key)

def chat_with_gpt(question):
    # Creating a conversation with the system message and user question
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ]

    # Make a request to the OpenAI API
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=conversation)

    # Extract the generated response
    answer = response.choices[0].message.content.strip()
    return answer

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
    """Create tables as per the updated schema."""
    user_profiles_table_sql = '''
    CREATE TABLE IF NOT EXISTS user_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER, 
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

    c = conn.cursor()
    c.execute(user_profiles_table_sql)
    c.execute(facial_embeddings_table_sql)

    conn.commit()


def insert_user_profile(conn, name="Anonymous", age=None):
    """Insert a user profile into the database with name and age."""
    try:
        # Encrypt sensitive data before insertion
        print(name)
        encrypted_name = encryption_tool.encrypt(name.encode())
        print(encrypted_name)
        encrypted_age = encryption_tool.encrypt(name.encode())
        sql = 'INSERT INTO user_profiles (name, age) VALUES (?, ?)'
        cur = conn.cursor()
        cur.execute(sql, (encrypted_name, encrypted_age))
        conn.commit()
        return cur.lastrowid
    except Exception as e:
        print(f"Error inserting data: {e}")


def insert_embedding(conn, user_id, embedding):
    """Insert facial embedding into the database."""
    try:
        # Convert the embedding tensor to a NumPy array and then to bytes
        embedding_bytes = embedding.tobytes()

        # Encrypt the embedding bytes
        encrypted_embedding_bytes = encryption_tool.encrypt(embedding_bytes)

        
        sql = "INSERT INTO facial_embeddings (user_id, embedding) VALUES (?, ?)"
        cur = conn.cursor()
        cur.execute(sql, (user_id, encrypted_embedding_bytes))
        conn.commit()
    except Exception as e:
        print(f"Error inserting embedding: {e}")

'''def insert_embedding(conn, user_id, embedding):
    """Insert facial embedding into the database."""
    try:
        # Convert the embedding tensor to a NumPy array and then to bytes
        embedding_bytes = embedding.tobytes()
    
        sql = "INSERT INTO facial_embeddings (user_id, embedding) VALUES (?, ?)"
        cur = conn.cursor()
        cur.execute(sql, (user_id, embedding_bytes))
        conn.commit()
    except Exception as e:
        print(f"Error inserting embedding: {e}")'''

        


def capture_embeddings_with_mediapipe(face_detection, facenet_model, image):
    """
    Detect faces using MediaPipe and capture facial embeddings using FaceNet.
    """
    # Convert the image color from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe Face Detection
    results = face_detection.process(image_rgb)
    
    embeddings = []
    if results.detections:
        for detection in results.detections:
            # Assuming you have a way to crop the face based on the detection bounding box
            # This is a placeholder to demonstrate the concept
            # You might need to adjust the bounding box coordinates and cropping logic
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                   int(bboxC.width * w), int(bboxC.height * h)
            cropped_face = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            face_image = Image.fromarray(cropped_face)
            face_image = face_image.resize((160, 160))
            face_tensor = transforms.ToTensor()(face_image).unsqueeze(0)
            
            # Generate the embedding using FaceNet model
            embedding = facenet_model(face_tensor)
            embeddings.append(embedding)
    
    return embeddings

def process_frames(face_detection, facenet_model, conn, user_name, user_age):
    user_id = insert_user_profile(conn, user_name, user_age)
    
    while not stop_event.is_set() or not frame_queue.empty():
        if not frame_queue.empty():
            image = frame_queue.get()
            embeddings = capture_embeddings_with_mediapipe(face_detection, facenet_model, image)
            if embeddings:
                for embedding in embeddings:
                    # Check if the embedding is a PyTorch tensor and requires grad
                    if isinstance(embedding, torch.Tensor) and embedding.requires_grad:
                        # Correct handling for PyTorch tensor by detaching and converting to NumPy array
                        numpy_embedding = embedding.detach().cpu().numpy()
                    else:
                        # If it's already a NumPy array or a tensor that doesn't require grad
                        numpy_embedding = embedding.cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding
                    insert_embedding(conn, user_id, numpy_embedding.flatten())
            
            frame_queue.task_done()

def get_user_consent(recognizer, microphone):
    """Ask the user for consent to process their facial features."""
    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            engine.say("Do you consent to have your facial features captured and analyzed for this session? Please say 'yes' or 'no'.")
            engine.runAndWait()
            audio = recognizer.listen(source)
        response = recognizer.recognize_google(audio).lower()
        
        if response == "yes":
            return True
        else:
            engine.say("You have not given consent to process your facial features. Exiting the application.")
            engine.runAndWait()
            return False
    except Exception as e:
        print(f"Error obtaining consent: {e}")
        return False


def get_user_name(recognizer, microphone):
    """Capture the user's name synchronously and return it."""
    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            engine.say("Please say your name after the beep.")
            engine.runAndWait()
            audio = recognizer.listen(source)
        name = recognizer.recognize_google(audio)
        print(f"Received name: {name}")
        name_phrase = name
        # Formulate the question
        chat = f"What is the name in the phrase '{name_phrase}'? Return just the name in your answer, nothing else."
        name = chat_with_gpt(chat)
        print("Chat Answer: ", name)
        return name
    except Exception as e:
        print(f"Error capturing name: {e}")
        return None
    
def get_user_age(recognizer, microphone):
    """Ask the user for their age and return it."""
    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            engine.say("Please tell me your age.")
            engine.runAndWait()
            audio = recognizer.listen(source)
        age = recognizer.recognize_google(audio)
        print(f"Received age: {age}")
        
        # Optional: Validate the received input to ensure it's a number
        try:
            age = int(age)  # Try converting the spoken age to an integer
            return age
        except ValueError:
            engine.say("I'm sorry, I didn't catch that. Could you please repeat your age?")
            engine.runAndWait()
            return None
    except Exception as e:
        print(f"Error capturing age: {e}")
        return None

def get_user_question(recognizer, microphone):
    """Ask the user to state their question."""
    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            engine.say("Please state your question now.")
            engine.runAndWait()
            audio = recognizer.listen(source)
        question = recognizer.recognize_google(audio)
        print(f"Received question: {question}")
        return question
    except Exception as e:
        print(f"Error capturing question: {e}")
        return None
    
def display_text_on_frame(frame, text, position=(50, 50), color=(0, 255, 0), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
    try:
        # Convert text to string
        text = str(text)
        # Convert color to tuple if not already
        if not isinstance(color, tuple):
            color = tuple(color)
        # Convert font to int if not already
        if not isinstance(font, int):
            font = getattr(cv2, font)
        cv2.putText(frame, text, position, font, font_scale, color, thickness)
    except Exception as e:
        print(f"Error displaying text on frame: {e}")

def main():
    db_file = 'encrypt.db'
    conn = create_connection(db_file)
    
    if conn is not None:
        create_tables(conn)
        cap = cv2.VideoCapture(0)
        last_snapshot_time = time.time()
        microphone = sr.Microphone()
        
        if not get_user_consent(recognizer, microphone):
            return  # Exit if consent not given

        user_name = get_user_name(recognizer, microphone)
        if user_name is None:
            print("Could not capture the user's name. Exiting.")
            return
        
        # After obtaining consent and the user's name
        user_age = get_user_age(recognizer, microphone)
        if user_age is None:
            print("Could not capture the user's age. Exiting.")
            return
        
        user_question = get_user_question(recognizer, microphone)
        if user_question is None:
            print("Could not capture the user's question. Exiting.")
            return
        
        print(f"Hello, {user_name}! You asked: {user_question}. Now starting to process video feed.")
        
        #user_name = get_user_name(recognizer, sr.Microphone())
        
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
            
            processing_thread = threading.Thread(target=process_frames, args=(face_detection, facenet_model, conn, user_name, user_age))
            processing_thread.start()

            try:
                while True:
                    ret, frame = cap.read()
                    if ret:
                                    # Convert frame to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
                        # Detect faces
                        results = face_detection.process(frame_rgb)
            
                        # Draw bounding box around each detected face
                        if results.detections:
                            for detection in results.detections:
                                bboxC = detection.location_data.relative_bounding_box
                                ih, iw, _ = frame.shape
                                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                    int(bboxC.width * iw), int(bboxC.height * ih)
                                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)#


                        #display_text_on_frame(frame, "Unknown User", position=(50, 50), color=(0, 255, 0))
                        
                        userInfo = f"Name: {user_name}, Age: {user_age}"
                        display_text_on_frame(frame, userInfo, position=(50, 50), color=(0, 255, 0))

                        cv2.imshow('Face Detection', frame)

                        current_time = time.time()
                        # Check if 5 seconds have passed since the last snapshot
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
