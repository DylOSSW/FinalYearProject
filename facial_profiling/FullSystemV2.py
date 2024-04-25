import cv2
import time
import threading
import queue
import speech_recognition as sr
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
from PIL import Image
import mediapipe as mp
import torch
import pygame
from database_operations_no_encryption import create_connection, create_tables, insert_user_profile, insert_embedding, delete_old_records, delete_database_on_exit
import pygame
import tempfile
import os
import speech_recognition as sr
from openai import OpenAI
import logging
import cProfile
import sys
import numpy as np


# Setup logging configuration
logging.basicConfig(filename='application_audit.log', level=logging.INFO, filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize the queue once, globally.
frame_queue = queue.Queue()
stop_event = threading.Event()
face_detected_event = threading.Event()
profile_mode_event = threading.Event()  # Event to signal profiling mode

def initialize_components():
    global frame_queue
    recognizer = sr.Recognizer()
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    pygame.mixer.init()
    stop_event = threading.Event()
    face_detected_event = threading.Event()
    audio_input_queue = queue.Queue()
    speech_processing_queue = queue.Queue()
    return recognizer, mp_face_detection, mp_drawing, stop_event, face_detected_event, audio_input_queue, speech_processing_queue

def start_profiling_thread(conn, cap, face_detection, frame_queue, stop_event, face_detected_event, profile_mode_event):
    # Reset the stop event in case it was set from a previous profiling session
    stop_event.clear()

    try:
        user_name = "New User"
        user_age = "Unknown"
        facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
        processing_thread = threading.Thread(target=process_frames, args=(face_detection, facenet_model, conn, user_name, user_age, stop_event, frame_queue))
        processing_thread.start()

        instructions = ["Please face forward for a few seconds.", "Now, please slowly turn to your left.", "And now, please slowly turn to your right."]
        for instruction in instructions:
            print(instruction)
            capture_for_duration(cap, frame_queue, duration=6)  # Passing frame_queue
        
        stop_event.set()
        processing_thread.join()
        print("Profiling done")

    finally:
        # Notify the main thread that profiling is done
        profile_mode_event.clear()

def process_frames(face_detection, facenet_model, conn, user_name, user_age, stop_event, frame_queue):

    user_id = insert_user_profile(conn, user_name, user_age)
    
    while not stop_event.is_set():

        if not frame_queue.empty():
            image = frame_queue.get()
            #print("took frame")
            embeddings = capture_embeddings_with_mediapipe(face_detection, facenet_model, image)
            if embeddings:
                for embedding in embeddings:
                    if isinstance(embedding, torch.Tensor) and embedding.requires_grad:
                        numpy_embedding = embedding.detach().cpu().numpy()
                    else:
                        numpy_embedding = embedding.cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding
                    print("Calling insert")
                    insert_embedding(conn, user_id, numpy_embedding.flatten())
            frame_queue.task_done()

def capture_embeddings_with_mediapipe(face_detection, facenet_model, image):
    """
    Detect faces using MediaPipe and capture facial embeddings using FaceNet.
    """
    # Process the image with MediaPipe Face Detection
    results = face_detection.process(image)
    
    embeddings = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                   int(bboxC.width * w), int(bboxC.height * h)
            cropped_face = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            face_image = Image.fromarray(cropped_face)
            face_image = face_image.resize((160, 160))
            face_tensor = transforms.ToTensor()(face_image).unsqueeze(0)
            
            # Disable gradient calculations
            with torch.no_grad():
                # Generate the embedding using FaceNet model
                embedding = facenet_model(face_tensor)
                embeddings.append(embedding)

    return embeddings

def get_all_embeddings(conn):

    """Retrieve all user embeddings from the database."""
    embeddings = []
    user_ids = []
    sql = "SELECT user_id, embedding FROM facial_embeddings"
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    for row in rows:
        user_ids.append(row[0])
        embedding = np.frombuffer(row[1], dtype=np.float32)
        embeddings.append(embedding)
    return user_ids, embeddings

def find_closest_embedding(captured_embedding, embeddings, threshold=0.9):
    """
    Find the closest embedding in the database to the captured one, with a threshold for matching.
    """
    min_distance = float('inf')
    closest_embedding_index = -1
    for i, db_embedding in enumerate(embeddings):
        # Ensure captured_embedding is a NumPy array
        if isinstance(captured_embedding, torch.Tensor):
            captured_embedding = captured_embedding.detach().cpu().numpy()

        # Calculate the distance
        distance = np.linalg.norm(captured_embedding - db_embedding)
        if distance < min_distance:
            min_distance = distance
            closest_embedding_index = i

    if min_distance > threshold:  # If no embedding is close enough, return no match
        return -1
    return closest_embedding_index


def face_detection_continuous(face_detection, cap, conn, face_detected_event, stop_event, profile_mode_event):
    user_ids, embeddings = get_all_embeddings(conn)
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
    face_detected_time = None
    recognition_frame_skip = 30
    frame_count = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)
            display_frame = frame.copy()

            if results.detections:
                if face_detected_time is None:
                    face_detected_time = time.time()
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih))
                    cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 2)

                if time.time() - face_detected_time >= 3 and not profile_mode_event.is_set():
                    frame_count += 1
                    if frame_count >= recognition_frame_skip:
                        frame_count = 0
                        # Attempt recognition
                        attempt_recognition(cap, face_detection, facenet_model, frame_rgb, embeddings, face_detected_event, user_ids, conn, profile_mode_event)
            else:
                face_detected_time = None  # Reset if no detections

            cv2.imshow('Face Recognition System', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    stop_event.set()


def capture_for_duration(cap, frame_queue, duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_queue.put(frame_rgb)  # Correct use of queue instance
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



def attempt_recognition(cap, face_detection, facenet_model, frame_rgb, embeddings, face_detected_event, user_ids, conn, profile_mode_event):
    embeddings_captured = capture_embeddings_with_mediapipe(face_detection, facenet_model, frame_rgb)
    if embeddings_captured:
        captured_embedding = embeddings_captured[0]
        index = find_closest_embedding(captured_embedding, embeddings)
        if index != -1:
            user_id = user_ids[index]
            print(f"Hello there, Recognized User ID {user_id}!")
            face_detected_event.set()
        else:
            print("User not recognized. Switching to profiling mode.")
            profile_mode_event.set()  # Trigger profiling mode
            # If not recognized, start profiling in a new thread
            profiling_thread = threading.Thread(target=start_profiling_thread, args=(conn, cap, face_detection, frame_queue, stop_event, face_detected_event, profile_mode_event))
            profiling_thread.start()

            
def main():
    recognizer, mp_face_detection, mp_drawing, stop_event, face_detected_event, audio_input_queue, speech_processing_queue = initialize_components()
    db_file = 'MYDB2.db'
    conn = create_connection(db_file)
    if conn:
        create_tables(conn)
        cap = cv2.VideoCapture(0)
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            face_detection_thread = threading.Thread(target=face_detection_continuous, args=(face_detection, cap, conn, face_detected_event, stop_event, profile_mode_event))
            face_detection_thread.start()
            face_detection_thread.join()
            conn.close()
            delete_database_on_exit(db_file)
            logging.info("Application exited")
    else:
        print("Failed to create a database connection.")

if __name__ == '__main__':
    main()