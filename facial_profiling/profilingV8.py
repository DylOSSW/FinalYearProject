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

# Setup logging configuration
logging.basicConfig(filename='application_audit.log', level=logging.INFO, filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Global list to keep track of temporary files
temp_files_to_delete = []

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
    response = client.audio.speech.create(model="tts-1", voice="onyx", input=text)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    with open(temp_file.name, 'wb') as file:
        file.write(response.content)
    # Add path to the global list for cleanup on exit
    temp_files_to_delete.append(temp_file.name)
    return temp_file.name

def speech_thread_function(recognizer, audio_input_queue, stop_event):
    with sr.Microphone() as source:
        while not stop_event.is_set():
            print("Listening...")
            audio = recognizer.listen(source)
            audio_input_queue.put(audio)

def process_audio_thread_function(recognizer, audio_input_queue, speech_processing_queue, stop_event):
    while not stop_event.is_set() or not audio_input_queue.empty():
        audio = audio_input_queue.get()
        try:
            text = recognizer.recognize_google(audio)
            speech_processing_queue.put(text)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
        audio_input_queue.task_done()

def play_audio(audio_file_path):
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
    
    pygame.mixer.music.load(audio_file_path)
    pygame.mixer.music.play()

    # Wait for the music to finish playing.
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    # Delaying the cleanup to ensure pygame has released the file
    pygame.mixer.music.unload()  # Ensure the mixer unloads the music to free the file.
    try:
        # Attempt to remove the file right after playback is confirmed to be stopped and unloaded
        if audio_file_path in temp_files_to_delete:
            os.remove(audio_file_path)
            temp_files_to_delete.remove(audio_file_path)  # Remove from cleanup list if successfully deleted
            print(f"Temp file {audio_file_path} deleted successfully.")
    except Exception as e:
        print(f"Error deleting temp file {audio_file_path} immediately: {e}. Will attempt later at exit.")


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
        logging.info("User consent received.")
        return True
    else:
        play_audio(synthesize_speech("You have not given consent to process your facial features. Exiting the application."))
        logging.info("User consent has not been given.")
        return False

def get_user_name(recognizer, microphone):
    name_response = get_user_input_with_retries(recognizer, microphone, "Please say your name after the beep.")
    if name_response:
        chat = f"What is the name in the phrase '{name_response}'? Return just the name in your answer, nothing else."
        name = chat_with_gpt(chat)
        print("Chat Answer: ", name)
        logging.info("User age has been received.")
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
        logging.info("User age has been received.")
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
        logging.info("User question has been captured.")
        return question_response
    else:
        print("Failed to capture the user's question.")
        play_audio(synthesize_speech("I'm sorry, I couldn't understand your question."))
        return None

def initialize_components():
    recognizer = sr.Recognizer()
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    pygame.mixer.init()
    frame_queue = queue.Queue()
    stop_event = threading.Event()
    face_detected_event = threading.Event()
    audio_input_queue = queue.Queue()
    speech_processing_queue = queue.Queue()
    return recognizer, mp_face_detection, mp_drawing, frame_queue, stop_event, face_detected_event, audio_input_queue, speech_processing_queue

'''def display_text_on_frame(frame, text, position=(50, 50), color=(0, 255, 0), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
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
        print(f"Error displaying text on frame: {e}")'''

def capture_embeddings_with_mediapipe(face_detection, facenet_model, image):
    """
    Detect faces using MediaPipe and capture facial embeddings using FaceNet.
    """
    # Convert the image color from BGR to RGB
    #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
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
            
            # Generate the embedding using FaceNet model
            embedding = facenet_model(face_tensor)
            embeddings.append(embedding)
    
    return embeddings

def process_frames(face_detection, facenet_model, conn, user_name, user_age, stop_event, frame_queue):

    user_id = insert_user_profile(conn, user_name, user_age)
    
    while not stop_event.is_set():
        if not frame_queue.empty():
            image = frame_queue.get()
            embeddings = capture_embeddings_with_mediapipe(face_detection, facenet_model, image)
            if embeddings:
                for embedding in embeddings:
                    if isinstance(embedding, torch.Tensor) and embedding.requires_grad:
                        numpy_embedding = embedding.detach().cpu().numpy()
                    else:
                        numpy_embedding = embedding.cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding
                    insert_embedding(conn, user_id, numpy_embedding.flatten())
            frame_queue.task_done()

def face_detection_continuous(face_detection, cap, stop_event, face_detected_event):
    face_detected_continuous_start = None
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            # Convert frame to RGB for face detection processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            # Always display the frame, even if no face is detected
            display_frame = frame.copy()

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                           int(bboxC.width * iw), int(bboxC.height * ih)
                    cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 2)
            
                name = "Unknown"
                age = "Unknown"
                userInfo = f"Name: {name}, Age: {age}"
                cv2.putText(display_frame, userInfo, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                modeText = "Mode: Profiling"  # Assuming 'Profile' is the string you mentioned
                cv2.putText(display_frame, modeText, (50, frame.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                if face_detected_continuous_start is None:
                    face_detected_continuous_start = time.time()
                elif time.time() - face_detected_continuous_start >= 3:
                    face_detected_event.set()
                    # Optionally, continue showing the video feed even after the face has been detected for 3 seconds
                    # You can remove the break statement if you want the detection and display to continue
                    # break
            else:
                face_detected_continuous_start = None

            cv2.imshow('Face Recognition System', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if not face_detected_event.is_set():
        print("No face detected for 3 seconds continuously. Exiting.")
        stop_event.set()

def capture_for_duration(cap, frame_queue, duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_queue.put(frame_rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Immediate exit option
            break

def main_interaction_process(conn, cap, face_detection, recognizer, frame_queue, stop_event, face_detected_event):
    if not face_detected_event.wait(timeout=140):  # Wait for a face to be detected continuously for 3 seconds
        print("Exiting due to no face detection.")
        return
    
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    # Play the greeting before asking for consent
    greeting_text = "Hello there, my name is Onyx."
    greeting_audio_path = synthesize_speech(greeting_text)
    play_audio(greeting_audio_path)

    if not get_user_consent(recognizer, microphone):
        print("Exiting due to lack of consent.")
        return

    user_name = get_user_name(recognizer, microphone)
    user_age = get_user_age(recognizer, microphone)
    #user_question = get_user_question(recognizer, microphone)

    #print(f"Hello, {user_name}! You asked: {user_question}. Now starting to process video feed.")

    
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

    # Start capturing frames continuously on a separate thread
    processing_thread = threading.Thread(target=process_frames, args=(face_detection, facenet_model, conn, user_name, user_age, stop_event, frame_queue))
    processing_thread.start()

        # Instructions
    instructions = [
        "Please face forward for a few seconds.",
        "Now, please slowly turn to your left.",
        "And now, please slowly turn to your right."
    ]

    # Play instructions and capture frames continuously
    for instruction in instructions:
        play_audio(synthesize_speech(instruction))
        # Capture frames for a duration after each instruction
        capture_for_duration(cap, frame_queue, duration=6)  # Adjust duration as needed

    #sys.exit()

def main():
    logging.info("Application started")
    recognizer, mp_face_detection, mp_drawing, frame_queue, stop_event, face_detected_event, audio_input_queue, speech_processing_queue = initialize_components()
    db_file = 'logging.db'
    conn = create_connection(db_file)
    if conn:
        create_tables(conn)
        cap = cv2.VideoCapture(0)
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            
            # Start a thread to periodically delete old records
            #deletion_thread = threading.Thread(target=delete_old_records, args=(conn,))
            #deletion_thread.daemon = True  # Set the thread as a daemon thread
            #deletion_thread.start()

            face_detection_thread = threading.Thread(target=face_detection_continuous, args=(face_detection, cap, stop_event, face_detected_event))
            speech_thread = threading.Thread(target=speech_thread_function, args=(recognizer, audio_input_queue, stop_event))
            process_audio_thread = threading.Thread(target=process_audio_thread_function, args=(recognizer, audio_input_queue, speech_processing_queue, stop_event))
            
            face_detection_thread.start()
            speech_thread.start()
            process_audio_thread.start()

            main_interaction_process(conn, cap, face_detection, recognizer, frame_queue, stop_event, face_detected_event)

            #stop_event.set()
            face_detection_thread.join()
            speech_thread.join()
            process_audio_thread.join()
            conn.close()
            delete_database_on_exit(db_file)
            logging.info("Application exited")
    else:
        print("Failed to create a database connection.")

if __name__ == '__main__':
    cProfile.run('main()', 'profiling_output.stats')
