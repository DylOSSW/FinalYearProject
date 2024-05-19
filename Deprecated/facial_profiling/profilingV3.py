import cv2
from datetime import datetime
import time
import threading
import queue
import speech_recognition as sr
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
import os
import mediapipe as mp
import torch
import tempfile
import pygame
from openai import OpenAI
from database_operations import create_connection, create_tables, insert_user_profile, insert_embedding

# Initialize speech recognition and text-to-speech engines
recognizer = sr.Recognizer()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
# Initialize pygame mixer
pygame.mixer.init()
# Global queue for frames
frame_queue = queue.Queue()
# Event to signal the processing thread to terminate
stop_event = threading.Event()

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

def synthesize_speech(text):
    global client  # Use the global client variable
    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_audio_file:
        tmp_audio_file.write(response.content)
        tmp_audio_file.flush()
    return tmp_audio_file.name


def play_audio(audio_file_path):
    try:
        pygame.mixer.music.load(audio_file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    finally:
        try:
            os.remove(audio_file_path)  # Attempt cleanup
        except Exception as e:
            print(f"Warning: Could not delete temp audio file '{audio_file_path}'. {e}")

def get_user_input_with_retries(recognizer, microphone, prompt, attempt_limit=3):
    attempts = 0
    while attempts < attempt_limit:
        try:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source)
                # Synthesize and play the prompt
                audio_file_path = synthesize_speech(prompt)
                play_audio(audio_file_path)
                # Listen for the user's response
                audio = recognizer.listen(source, timeout=5.0)  # Adjust timeout as necessary
            response = recognizer.recognize_google(audio).lower()
            return response
        except sr.WaitTimeoutError:  # Handling case where no speech is detected within the timeout
            attempts += 1
            print("I didn't catch that. Let's try again.")
            if attempts >= attempt_limit:
                play_audio(synthesize_speech("Sorry, I couldn't hear anything."))
        except sr.UnknownValueError:  # Handling case where speech is detected but not recognized
            attempts += 1
            print("I couldn't understand that. Let's try again.")
            if attempts >= attempt_limit:
                play_audio(synthesize_speech("Sorry, I couldn't understand you."))
        except Exception as e:
            print(f"An error occurred: {e}")
            play_audio(synthesize_speech("An error occurred. Let's try again."))
            attempts += 1
            if attempts >= attempt_limit:
                play_audio(synthesize_speech("Sorry, I couldn't process your input after several attempts."))
                break
    return None


def get_user_consent(recognizer, microphone):
    consent_response = get_user_input_with_retries(recognizer, microphone, 
                                                   "Do you consent to have your facial features captured and analyzed for this session? Please say 'yes' or 'no'.")
    if consent_response == "yes":
        play_audio(synthesize_speech("Thank you for your consent."))
        return True
    else:
        play_audio(synthesize_speech("You have not given consent to process your facial features. Exiting the application."))
        return False

def get_user_name(recognizer, microphone):
    name_response = get_user_input_with_retries(recognizer, microphone, "Please say your name after the beep.")
    if name_response:
        chat = f"What is the name in the phrase '{name_response}'? Return just the name in your answer, nothing else."
        name = chat_with_gpt(chat)
        print("Chat Answer: ", name)
        play_audio(synthesize_speech(f"Thank you, {name}."))
        return name
    else:
        print("Failed to capture the user's name.")
        return None

def get_user_age(recognizer, microphone):
    age_response = get_user_input_with_retries(recognizer, microphone, "Please tell me your age.")
    try:
        age = int(age_response)
        play_audio(synthesize_speech(f"Thank you. I have recorded your age as {age}."))
        return age
    except ValueError:
        play_audio(synthesize_speech("The provided age did not seem to be a valid number. Let's try again."))
        return None
    except Exception as e:
        print(f"Error capturing age: {e}")
        return None
    
def get_user_question(recognizer, microphone):
    question_response = get_user_input_with_retries(recognizer, microphone, "Please state your question now.")
    if question_response:
        play_audio(synthesize_speech("Thank you. I have recorded your question."))
        return question_response
    else:
        print("Failed to capture the user's question.")
        play_audio(synthesize_speech("I'm sorry, I couldn't understand your question."))
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
            print("Exiting due to lack of consent.")
            return

        user_name = get_user_name(recognizer, microphone)
        if user_name is None:
            print("Could not capture the user's name. Exiting.")
            return

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
