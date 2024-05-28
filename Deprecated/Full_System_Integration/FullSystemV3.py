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
from FullSystemV3_imports import live_speech_to_text, get_user_consent_for_recognition_attempt, play_audio
import pygame
import tempfile
import os
import speech_recognition as sr
from openai import OpenAI
import logging
import cProfile
import sys
import numpy as np
import audioop
import pyaudio
import wave
import os
import numpy as np
from playsound import playsound
import psutil
import GPUtil

# Setup logging configuration
logging.basicConfig(filename='application_audit.log', level=logging.INFO, filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize the queue once, globally.
frame_queue = queue.Queue()

stop_event = threading.Event()
face_detected_event = threading.Event()             # Face Detected Event: Triggers when the system detects a face.
has_profile_event = threading.Event()               # Has Profile Event: User indicates they have a profile.
does_not_have_profile_event = threading.Event()     # No Profile Event: User indicates they do not have a profile.  
profile_completed_event = threading.Event()         # Profile Completed Event: Profile collection is completed.
recognition_success_event = threading.Event()       # Recognition Success Event: User is successfully recognized.
recognition_failure_event = threading.Event()       # Recognition Failure Event: User is not recognized.
conversation_ended_event = threading.Event()        # Conversation Ended Event: Conversation has ended.

def initialize_components():
    global frame_queue
    mp_face_detection = mp.solutions.face_detection
    pygame.mixer.init()
    stop_event = threading.Event()
    face_detected_event = threading.Event()
    profile_mode_event = threading.Event()  # Ensure this is a threading.Event
    speech_processing_queue = queue.Queue()
    return  mp_face_detection,  stop_event, face_detected_event, profile_mode_event, speech_processing_queue

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

def capture_for_duration(cap, frame_queue, duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_queue.put(frame_rgb)  # Correct use of queue instance
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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


def attempt_recognition(cap, face_detection, frame_rgb, embeddings, face_detected_event, user_ids, conn, profile_mode_event):
    recognition_count = 0  # Variable to count successful recognitions
    
    while True:
        facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
        embeddings_captured = capture_embeddings_with_mediapipe(face_detection, facenet_model, frame_rgb)
        if embeddings_captured:
            captured_embedding = embeddings_captured[0]
            index = find_closest_embedding(captured_embedding, embeddings)
            if index != -1:
                user_id = user_ids[index]
                print(f"Hello there, Recognized User ID {user_id}!")
                recognition_count += 1  # Increment recognition count
                if recognition_count >= 3:
                    print("Three successful recognitions. Starting Conversation.")
                    #face_detected_event.set()  # Set face detected event
                    recognition_success_event.set()
                    break  # Exit the loop after successful recognition
                else:
                    print(f"{3 - recognition_count} more recognitions needed for event.")
            else:
                print("User not recognized. Switching to profiling mode.")
                recognition_failure_event.set()      # Recognition Failure Event: User is not recognized.
                break  # Exit the loop after failed recognition

# Rest of the code remains the same

def start_profiling_thread(conn, cap, face_detection, frame_queue):
    # Reset the stop event in case it was set from a previous profiling session
    stop_event.clear()

    try:
        '''if not get_user_consent_for_profiling():
            print("Exiting due to lack of consent.")
            return'''
        
        #user_name = get_user_name()
        #user_age = get_user_age()
        user_name = "Dylan"
        user_age = 26
        facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
        processing_thread = threading.Thread(target=process_frames, args=(face_detection, facenet_model, conn, user_name, user_age, stop_event, frame_queue))
        processing_thread.start()

        instructions = ["Please face forward for a few seconds.", "Now, please slowly turn to your left.", "And now, please slowly turn to your right."]
        for instruction in instructions:
            play_audio(instruction)
            capture_for_duration(cap, frame_queue, duration=2)
        
        stop_event.set()
        processing_thread.join()
        print("Profiling done")

    finally:
        # Notify the main thread that profiling is done
        profile_completed_event.set()

def check_profile_state():

    #response = get_user_input_with_retries(recognizer, microphone, "Do you have a profile? Say 'yes' or 'no'.")
    response  = input("Do you have a profile?: ")
        
    if response.lower() == 'yes':
         has_profile_event.set()
         print("User confirmed having a profile.")
    elif response.lower() == 'no':
         does_not_have_profile_event.set()
         print("User confirmed not having a profile.")


def conversation(conn, cap, face_detection, frame_queue):
    print("Starting conversation...")
    while True:
        # Simulate user input.
        user_input = input("User: ")

        # Dummy logic to respond based on user input
        if "hello" in user_input.lower():
            print("Bot: Hello there!")
        elif "how are you" in user_input.lower():
            print("Bot: I'm just a program, but thanks for asking!")
        elif "bye" in user_input.lower():
            print("Bot: Goodbye!")
            break  # End the conversation loop
        else:
            print("Bot: Sorry, I didn't understand that.")

        # Simulate a pause between user input and bot response
        time.sleep(1)

    print("Conversation ended.")
    conversation_ended_event.set()


def main():
    mp_face_detection, stop_event, face_detected_event, profile_mode_event, speech_processing_queue = initialize_components()
    db_file = 'MYDB2.db'
    conn = create_connection(db_file)
    speech_thread = threading.Thread(target=live_speech_to_text, args=(speech_processing_queue,))
    speech_thread.start()
    if conn:
        create_tables(conn)
        cap = cv2.VideoCapture(0)
        try:
            with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
                user_ids, embeddings = get_all_embeddings(conn)
                face_detected_time = None

                while not stop_event.is_set():
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = face_detection.process(frame_rgb)
                        display_frame = frame.copy()

                        if results.detections:
                            if face_detected_time is None:
                                face_detected_time = time.time()  # Start the timer on the first detection

                            if time.time() - face_detected_time >= 3:  # Check if the face has been detected continuously for 3 seconds
                                if not face_detected_event.is_set():  # Check this only once
                                    face_detected_event.set()
                                    print("Face detected for 3 seconds, initiating check profile state")
                                    threading.Thread(target=check_profile_state).start()
                                    face_detected_time = None  # Reset timer after action

                            if does_not_have_profile_event.is_set():
                                print("Starting profiling mode.")
                                profile_thread = threading.Thread(target=start_profiling_thread, args=(conn, cap, face_detection, frame_queue))
                                profile_thread.start()
                                does_not_have_profile_event.clear()  # Reset after starting profiling

                            if has_profile_event.is_set():
                                print("Attempting Recognition.")
                                recognition_thread = threading.Thread(target=attempt_recognition, args=(cap, face_detection, frame_rgb, embeddings, face_detected_event, user_ids, conn, profile_mode_event))
                                recognition_thread.start()
                                has_profile_event.clear()  # Reset after starting profiling

                            if recognition_failure_event.is_set():
                                print("Attempting Recognition.")
                                recognition_thread.join()
                                profile_from_recognition_thread = threading.Thread(target=start_profiling_thread, args=(conn, cap, face_detection, frame_queue))
                                profile_from_recognition_thread.start()
                                recognition_failure_event.clear()  # Reset after starting profiling

                            if recognition_success_event.is_set():
                                print("Attempting Recognition.")
                                recognition_thread.join()
                                conversation_thread = threading.Thread(target=conversation, args=(conn, cap, face_detection, frame_queue))
                                conversation_thread.start()
                                recognition_success_event.clear()  # Reset after starting profiling

                            if profile_completed_event.is_set():
                                profile_thread.join()
                                print("Profile completed. Ready for new face detection.")
                                face_detected_time = None  # Reset to detect new face
                                face_detected_event.clear()  # Allow new face detection
                                profile_completed_event.clear()  # Reset profiling event

                            
                            if conversation_ended_event.is_set():
                                conversation_thread.join()
                                print("Conversation completed. Ready for new face detection.")
                                face_detected_time = None  # Reset to detect new face
                                face_detected_event.clear()  # Allow new face detection
                                conversation_ended_event.clear()  # Reset profiling event

                            # Drawing bounding boxes and other UI updates here...

                        else:
                            face_detected_time = None  # Reset the timer if no face is detected

                                                # Example of getting operational info
                        thread_count = threading.active_count()


                        # Initialize vertical position offset
                        vertical_pos = frame.shape[0] - 50  # Starting from bottom, going up
                        line_height = 20  # Height of each line of text

                        # Draw Threads info
                        cv2.putText(display_frame, f"Threads: {thread_count}", 
                            (10, vertical_pos), cv2.FONT_HERSHEY_COMPLEX, 
                                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        vertical_pos = min(frame.shape[0] - 50, 50)  # Starting from bottom, going up, but ensuring at least 50 pixels from the bottom

                        # Mode text and other UI elements
                        cv2.imshow('User View', display_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
 
                stop_event.set()
        finally:
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            conn.close()
            #delete_database_on_exit(db_file)
            logging.info("Application exited")
    else:
        print("Failed to create a database connection.")

if __name__ == '__main__':
    main()
