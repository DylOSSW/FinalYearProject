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
import tempfile
import os
from openai import OpenAI
import logging
import cProfile
import sys
#from playsound import playsound
import audioop
import pyaudio
import wave
import os
import numpy as np
from playsound import playsound



last_matched_user_id = None
last_matched_time = time.time()
#info_display_timeout = 5  # seconds

display_info = {"name": "Unknown", "age": "Unknown"}
display_timeout = 5  # seconds to wait before clearing display if no new match
last_display_update = time.time()  # track last update to display info


# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Global queue for frames
recognition_frame_queue = queue.Queue()

profiling_frame_queue = queue.Queue()
# Event to signal the processing thread to terminate
stop_event = threading.Event()
# This is a new global queue for user info messages.
user_info_queue = queue.Queue()

MAX_FRAMES_TO_CAPTURE = 1

profiling_needed = False

new_profile = True

ambient_detected = False
speech_volume = 100
listening_enabled = False  # Flag to control the listening process

audio_input_queue = queue.Queue()

# Setup logging configuration
logging.basicConfig(filename='application_audit.log', level=logging.INFO, filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

audio_files = {
    "Hello there my name is Onyx.": "greeting_text.mp3",
    "I didn't catch that. Let's try again.": "didnt_catch_that.mp3",
    "Sorry, I couldn't hear anything.": "couldnt_hear_anything.mp3",
    "I couldn't understand that. Let's try again.": "couldnt_understand_that.mp3",
    "Sorry, I couldn't understand you.": "couldnt_understand_you.mp3",
    "An error occurred. Let's try again.": "error_occurred_try_again.mp3",
    "Sorry, I couldn't process your input after several attempts.": "couldnt_process_input.mp3",
    "Do you consent to have your facial features captured and analyzed for this session? Please say 'yes' or 'no'.": "ask_consent.mp3",
    "Thank you for your consent.": "thank_you_for_consent.mp3",
    "You have not given consent to process your facial features. Exiting the application.": "no_consent_exiting.mp3",
    "Please say your name.": "ask_name.mp3",
    "Thank you for providing your name": "thank_you_name.mp3",
    "Failed to capture the user's name.": "failed_name_capture.mp3",
    "Please tell me your age.": "ask_age.mp3",
    "Thank you. I have recorded your age.": "age_recorded.mp3",
    "I couldn't understand your age. Let's try again.": "couldnt_understand_age.mp3",
    "Failed to capture the user's age.": "failed_age_capture.mp3",
    "I'm sorry, I couldn't hear you clearly.": "couldnt_hear_clearly.mp3",
    "Please state your question now.": "ask_question.mp3",
    "Thank you. I have recorded your question.": "question_recorded.mp3",
    "I'm sorry, I couldn't understand your question.": "couldnt_understand_question.mp3",
    "Failed to capture the user's question.": "failed_question_capture.mp3",
    "Please face forward for a few seconds.": "face_forward.mp3",
    "Now, please slowly turn to your left.": "turn_left.mp3",
    "And now, please slowly turn to your right.": "turn_right.mp3"
}

def initialize_components():
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    frame_queue = queue.Queue()
    stop_event = threading.Event()
    face_detected_event = threading.Event()
    #audio_input_queue = queue.Queue()
    start_profiling_event = threading.Event()  # Add this line
    return mp_face_detection, mp_drawing, frame_queue, stop_event, face_detected_event, start_profiling_event
#, audio_input_queue,


def load_openai_key():
    with open('openai_api.key', 'rb') as key_file:
        key = key_file.read().decode('utf-8')  # Decode bytes to string
    return key

api_key = load_openai_key()

# Initialize the OpenAI client with the api_key as a named (keyword) argument
client = OpenAI(api_key=api_key)

def chat_with_gpt(question):
    global listening_enabled
    # Creating a conversation with the system message and user question
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ]

    listening_enabled = False

    # Make a request to the OpenAI API
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=conversation)

    # Extract the generated response
    answer = response.choices[0].message.content.strip()
    return answer



# Function to generate audio using OpenAI's TTS
def generate_audio(response_from_gpt,audio_file_path):
    audio_response = client.audio.speech.create(input=response_from_gpt, voice="onyx",model='tts-1')
    audio_response.stream_to_file(audio_file_path)
    

# Function to play the audio file using playsound
def play_response(audio_file_path):
    playsound(audio_file_path)


def play_audio(message_key, name=None):
    global listening_enabled
    try:
        if listening_enabled:
            listening_enabled = False
        # Handle dynamic insertion for name-specific messages
        if name and '{name}' in message_key:
            message_key = message_key.format(name=name)

        audio_file_path = 'prompts/' + audio_files[message_key]
        print(f"Playing audio: {audio_file_path}")
        
        # Load and play the audio file using playsound
        playsound(audio_file_path)
        
    except Exception as e:
        print(f"Error playing audio file {audio_file_path}: {e}")
    listening_enabled = True
      
# Function to perform live speech-to-text
def live_speech_to_text(audio_input_queue,wait_time=70):
    global ambient_detected
    global speech_volume
    global listening_enabled
    
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024

    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    frames = []
    recording = False
    frames_recorded = 0

    while True:
        #print(listening_enabled)
        frames_recorded += 1
        data = stream.read(CHUNK)
        rms = audioop.rms(data, 2)

        if not ambient_detected:
            if frames_recorded < 40:
                if frames_recorded == 1:
                    print("Detecting ambient noise...")
                if frames_recorded > 5:
                    if speech_volume < rms:
                        speech_volume = rms
                continue
            elif frames_recorded == 40:
                print("Listening...")
                speech_volume = speech_volume * 3
                ambient_detected = True

        if rms > speech_volume and listening_enabled:
            recording = True
            print("Recording ENABLED")
            frames_recorded = 0
        elif recording and frames_recorded > wait_time:
            recording = False
            

            wf = wave.open("audio.wav", 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            audio_file = open("audio.wav", "rb")
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            
            audio_file.close()  # Close the file after usage

            os.remove("audio.wav")

            print("Result org: ",result)
            print("Result text: ",result.text)
            audio_input_queue.put(result.text)
            frames = []

        if recording:
            frames.append(data)

# Function to process audio data from the queue - Conversation piece**
#Set flag for listening_enabled
def process_audio_data(audio_input_queue):
    global listening_enabled 

    while True:
        #listening_enabled = True
        try:
            text = audio_input_queue.get()
            print("Text from audio queue: ", text)
            response_from_gpt = chat_with_gpt(text)
            audio_file_path = "temp_audio.mp3"  # Temporary file path for the audio
            generate_audio(response_from_gpt, audio_file_path)
            if audio_file_path is not None:
                play_response(audio_file_path)
                #playsound(None)  # Close the audio player
                os.remove(audio_file_path)  # Remove the temporary audio file
            else:
                print("Error: Audio file path is None")
            listening_enabled = True

        except PermissionError as pe:
            print(f"Permission denied error: {pe}")

        except Exception as e:
            print(f"Error in audio processing: {e}")

def get_user_input_with_retries(prompt_key, attempt_limit=3):
    global listening_enabled
    attempts = 0
    while attempts < attempt_limit:
        if not listening_enabled:
            listening_enabled = True
        try:
            # Play the prompt from a pre-recorded file
            play_audio(prompt_key)
            try:
                response = audio_input_queue.get()
                
                return response
            except queue.Empty:
                attempts += 1
                play_audio("I didn't catch that. Let's try again.")
                if attempts == 2:
                    attempts += 1
                    play_audio("I couldn't understand that. Let's try again.")
                elif attempts >= attempt_limit:
                    play_audio("Sorry, I couldn't hear anything.")
        except Exception as e:
            attempts += 1
            play_audio("An error occurred. Let's try again.")
            if attempts >= attempt_limit:
                play_audio("Sorry, I couldn't process your input after several attempts.")
    listening_enabled = False
    return None

def get_user_consent():
    consent_response = get_user_input_with_retries("Do you consent to have your facial features captured and analyzed for this session? Please say 'yes' or 'no'.")
    print("Consent response.lower(): ", consent_response.lower())
    if "yes" in consent_response.lower():
        play_audio("Thank you for your consent.")
        logging.info("User consent received.")
        return True
    else:
        play_audio("You have not given consent to process your facial features. Exiting the application.")
        logging.info("User consent has not been given.")
        return False

def get_user_name():
    name_response = get_user_input_with_retries("Please say your name.")
    if name_response:
        chat = f"What is the name in the phrase '{name_response}'? Return just the name in your answer, nothing else."
        name = chat_with_gpt(chat)
        print("Chat Answer: ", name)
        logging.info("User name has been received.")
        play_audio("Thank you for providing your name")  # Handling dynamic insertion
        return name
    else:
        print("Failed to capture the user's name.")
        return None

def get_user_age():
    age_response = get_user_input_with_retries("Please tell me your age.")
    if age_response:
        chat = f"How old is someone who says, '{age_response}'? Return just the age in your answer."
        age_text = chat_with_gpt(chat)
        try:
            age = int(age_text)
            print("Extracted Age: ", age)
            logging.info("User age has been received.")
            play_audio("Thank you. I have recorded your age.")
            return age
        except ValueError:
            print("Failed to extract a valid age.")
            play_audio("I couldn't understand your age. Let's try again.")
            return None
    else:
        print("Failed to capture the user's age.")
        play_audio("I'm sorry, I couldn't hear you clearly.")
        return None
 
def get_user_question():
    question_response = get_user_input_with_retries("Please state your question now.")
    if question_response:
        play_audio("Thank you. I have recorded your question.")
        logging.info("User question has been captured.")
        return question_response
    else:
        print("Failed to capture the user's question.")
        play_audio("I'm sorry, I couldn't understand your question.")
        return None


def capture_embeddings(face_detection, facenet_model, image):
    print("----ENTERED CAPTURE EMBEDDINGS----")
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
            
            # Generate the embedding using FaceNet model
            print("---GENERATING EMBEDDING---")
            embedding = facenet_model(face_tensor)
            print("---FACENET EMBEDDING GENERATED---")
            #embeddings.append(embedding)
            embeddings.append(embedding.cpu().detach().numpy().flatten())
            print("----EMBEDDING APPENDED: EXITING CAPTURE EMBEDDINGS----")


    return embeddings

def process_frames(face_detection, facenet_model, conn, user_name, user_age, stop_event, start_profiling_event):

    user_id = insert_user_profile(conn, user_name, user_age)
    
    while not stop_event.is_set():
        print("---TOP OF WHILE LOOP IN PROFILING PROCESS FRAMES---")
        #start_profiling_event.wait()
        image = profiling_frame_queue.get()
        embeddings = capture_embeddings(face_detection, facenet_model, image)
        if embeddings:
            for embedding in embeddings:
                if isinstance(embedding, torch.Tensor) and embedding.requires_grad:
                    numpy_embedding = embedding.detach().cpu().numpy()
                else:
                    numpy_embedding = embedding.cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding
                insert_embedding(conn, user_id, numpy_embedding.flatten())
                print("embedding inserted")
        profiling_frame_queue.task_done()
        print("---TASK DONE IN PROCESSING FRAMES FOR PROFILING---")

def capture_for_duration(cap, frame_queue, duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        pass
        #ret, frame = cap.read()
        #if ret:
        #    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #    frame_queue.put(frame_rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Immediate exit option
            break

def get_all_embeddings(conn):
    print("---INSIDE GET ALL EMBEDDINGS---")

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
    print("---RETURNING IDS AND EMBEDDINGS FROM GET ALL EMBEDDINGS---")
    return user_ids, embeddings

def find_closest_embedding(captured_embedding, embeddings, threshold=0.6):
    print("---INSIDE FIND CLOSEST EMBEDDING---")
    """Find the closest embedding in the database to the captured one, with a threshold for matching."""
    min_distance = float('inf')
    closest_embedding_index = -1
    for i, db_embedding in enumerate(embeddings):
        distance = np.linalg.norm(captured_embedding - db_embedding)
        if distance < min_distance:
            min_distance = distance
            closest_embedding_index = i
    if min_distance > threshold:  # If no embedding is close enough, return no match
        return -1
    print("---BOTTOM OF FINDING CLOSEST EMBEDDING ON SUCCESS---")
    return closest_embedding_index



def main_profiling_process(conn, cap, face_detection, frame_queue, stop_event, face_detected_event, start_profiling_event):
    global profiling_needed,new_profile, listening_enabled
    while not stop_event.is_set():
        start_profiling_event.wait()  # This will wait indefinitely until the event is set
        
        print("Event caught - starting profiling")
        #print("profiling thread running")
        #if profiling_needed:
         #   listening_enabled = True
          #  print("INSIDE PROFILING")
            #if not face_detected_event.wait(timeout=140):  # Wait for a face to be detected continuously for 3 seconds
            #   print("Exiting due to no face detection.")
            #  return
            
            
            

        # Play the greeting before asking for consent
        play_audio("Hello there my name is Onyx.")

        if not get_user_consent():
            print("Exiting due to lack of consent.")
            return

        user_name = get_user_name()
        user_age = get_user_age()
        #user_question = get_user_question(recognizer, microphone)

        #print(f"Hello, {user_name}! You asked: {user_question}. Now starting to process video feed.")

        
        facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

        # Start capturing frames continuously on a separate thread
        processing_thread = threading.Thread(target=process_frames, args=(face_detection, facenet_model, conn, user_name, user_age, stop_event, start_profiling_event))
        processing_thread.start()

            # Instructions
        instructions = [
        "Please face forward for a few seconds.",
        "Now, please slowly turn to your left.",
        "And now, please slowly turn to your right."
        ]

        # Play instructions and capture frames continuously
        for instruction in instructions:
            print(instruction)
            #play_audio(instruction)
            time.sleep(12)
            # Capture frames for a duration after each instruction
            #capture_for_duration(cap, frame_queue, duration=6)  # Adjust duration as needed
            
        print("finished")
        listening_enabled = False
        new_profile = True
        start_profiling_event.clear()  # Clear the event after catching it
        processing_thread.join()
        #profiling_needed = False
        
    #sys.exit()

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

def recognition_process_frames(detector, facenet_model, conn, start_profiling_event, face_detected_event):
    print("---INSIDE RECOGNITION FRAME PROCESSING---")
    global last_matched_user_id, last_matched_time, new_profile
    user_ids, embeddings = get_all_embeddings(conn)
    retry_max = 3
    retry_counter = 0
    match_threshold = 1
    while not stop_event.is_set():
        face_detected_event.wait()
        #print("---RECOGNITION ACTIVE: FACE DETECTED EVENT IS SET---")
        
        
    
        if new_profile:
            print("---GETTING UPDATED EMBEDDINGS: NEW PROFILE FOUND")
            new_profile = False
            user_ids, embeddings = get_all_embeddings(conn)
        if not recognition_frame_queue.empty():
            #image = frame_queue.get()
            #captured_embeddings = capture_embeddings(detector, facenet_model, image)
            captured_frames = []
            matched_frame_indexes = []  # Store indexes of frames with matches
            matched_user_index = []
            for _ in range(MAX_FRAMES_TO_CAPTURE):
                if not recognition_frame_queue.empty():
                    captured_frames.append(recognition_frame_queue.get())
                    
                
            # Capture embeddings for each frame
            captured_embeddings = []
            for frame in captured_frames:
                captured_embeddings.extend(capture_embeddings(detector, facenet_model, frame))

            if captured_embeddings:
                #Count positive matches
                num_matches = 0
                for index, captured_embedding in enumerate(captured_embeddings):
                    captured_embedding_np = np.array(captured_embedding).flatten()
                    closest_index = find_closest_embedding(captured_embedding_np, embeddings)
                    if closest_index != -1:
                        retry_counter = 0
                        num_matches += 1
                        matched_frame_indexes.append(index)  # Store index of matching frame
                        matched_user_index.append(closest_index)
                if num_matches >= match_threshold:
                    #retry_counter = 0
                    print("+++Match found+++")
                    print("Matched Frame Indexes: ",matched_frame_indexes)
                    print("Matched closest user indexes: ", matched_user_index)
                    
                    # Use the index from one of the matching frames to get closest index
                    closest_index = matched_user_index[0]  # Use the first matched index directly

                    current_time = time.time()
                    if last_matched_user_id != user_ids[closest_index]:
                        print("Last matched user id doesnt match known users")
                        #print("Last match id: ", last_matched_user_id, "---", closest_index, "\n",user_ids)
                        user_id = user_ids[closest_index]
                        last_matched_user_id = user_ids[closest_index]
                        last_matched_time = current_time
                        
                        # Fetch user details from the database
                        cur = conn.cursor()
                        cur.execute("SELECT name, age FROM user_profiles WHERE id = ?", (user_id,))
                        result = cur.fetchone()
                        if result:
                            name, age = result
                            print(f"Match found: Name - {name}, Age - {age}")  # Print match details
                            user_info_queue.put(result)  # Update the queue with new user info
                else:
                    retry_counter+=1
                    print(f"---Match NOT found---\nRetries left: {retry_max - retry_counter}")
                    
                    if retry_counter == retry_max:
                        retry_counter = 0
                        retry_max = 3
                        current_time = time.time()
                        # No match found, clear the display after a timeout
                        if last_matched_user_id is not None: # and (current_time - last_matched_time > info_display_timeout):
                            last_matched_user_id = None
                        
                        last_matched_time = current_time
                        print("No match found - Need to profile.")  # Print no match
                        user_info_queue.put(("Unknown", "Unknown"))
                        #profiling_needed = True
                        #print(f"Set profiling_needed to true: {profiling_needed}")
                        start_profiling_event.set()  # Signal that profiling should start
                        print("No match found - Start profiling.")
            recognition_frame_queue.task_done()
            print("---COMPARISON COMPLETE---")
            #frame.queue.pop()            

def main():
    global last_matched_user_id, last_matched_time, display_info, last_display_update
    face_detected_continuous_start = None
    face_detected_recognition = None
    face_detected_profiling = None
    logging.info("Application started")
    mp_face_detection, mp_drawing, frame_queue, stop_event, face_detected_event,start_profiling_event = initialize_components()
    #audio_input_queue
    db_file = 'logging.db'
    conn = create_connection(db_file)
    if conn:
        create_tables(conn)
        cap = cv2.VideoCapture(0)
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
            
            # Start a thread to periodically delete old records
            #deletion_thread = threading.Thread(target=delete_old_records, args=(conn,))
            #deletion_thread.daemon = True  # Set the thread as a daemon thread
            #deletion_thread.start()

            #face_detection_thread = threading.Thread(target=face_detection_continuous, args=(face_detection, cap, stop_event, face_detected_event))
            speech_thread = threading.Thread(target=live_speech_to_text, args=(audio_input_queue,))
            #process_audio_thread = threading.Thread(target=process_audio_data, args=(audio_input_queue,))
            
            #face_detection_thread.start()
            speech_thread.start()
            #process_audio_thread.start()

            # Start the recognition thread
            recognition_thread = threading.Thread(target=recognition_process_frames, args=(face_detection, facenet_model, conn, start_profiling_event, face_detected_event))
            recognition_thread.start()

            # Start the profiling thread
            profiling_thread = threading.Thread(target=main_profiling_process, args=(conn, cap, face_detection, profiling_frame_queue, stop_event, face_detected_event, start_profiling_event))
            profiling_thread.start()

        
            try:
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
                        
                            # Check for face detection for recognition
                            if face_detected_recognition is None:
                                face_detected_recognition = time.time()
                            elif time.time() - face_detected_recognition >= 3:
                                if not start_profiling_event.is_set() and recognition_frame_queue.empty():
                                    # Capture multiple frames
                                    captured_frames = []
                                    for _ in range(MAX_FRAMES_TO_CAPTURE):
                                        captured_frames.append(frame)
                                    # Put captured frames into the queue for processing
                                    for index, frame in enumerate(captured_frames):
                                        recognition_frame_queue.put(frame)
                                        print(f"---Recognition Frame Nr: {index} In Recognition Queue---")
                                    face_detected_recognition = time.time()
                            if face_detected_profiling is None:
                                    face_detected_profiling = time.time()
                            elif time.time() - face_detected_profiling >= 0:
                                if start_profiling_event.is_set() and profiling_frame_queue.empty():
                                #print(f"In detection - profiling has returned and is True")
                                    #print("Face detected for profiling for 4 seconds")
                                    #if profiling_frame_queue.empty():
                                    captured_frames = []
                                    for _ in range(MAX_FRAMES_TO_CAPTURE):
                                        captured_frames.append(frame)
                                    for index, frame in enumerate(captured_frames):
                                        profiling_frame_queue.put(frame)
                                        print(f"---Profiling frame Nr: {index} In Profiling Queue---")
                                    face_detected_profiling = time.time()
                            # Update display information based on recognition or profiling mode
                            if not user_info_queue.empty():
                                user_name, user_age = user_info_queue.get()
                                display_info["name"] = user_name
                                display_info["age"] = user_age
                        

                            modeText = "Mode: Profiling"  # Assuming 'Profile' is the string you mentioned
                            cv2.putText(display_frame, modeText, (50, frame.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            display_text = f"Name: {display_info['name']}, Age: {display_info['age']}"
                            cv2.putText(display_frame, display_text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            if face_detected_continuous_start is None:
                            #if not profiling_needed and face_detected_continuous_start is None:
                                face_detected_continuous_start = time.time()
                            elif time.time() - face_detected_continuous_start >= 3:
                                face_detected_event.set()
                        else:
                            face_detected_continuous_start = None
                            face_detected_recognition = None
                            face_detected_profiling = None
                            face_detected_event.clear()

                        cv2.imshow('Face Recognition System', display_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        continue

            finally:
                stop_event.set()
                #processing_thread.join()
                conn.close()
                cap.release()
                cv2.destroyAllWindows()
            #stop_event.set()
            #face_detection_thread.join()
                speech_thread.join()
                #process_audio_thread.join()
                recognition_thread.join()
                profiling_thread.join()
                conn.close()
                delete_database_on_exit(db_file)
                logging.info("Application exited")
    else:
        print("Failed to create a database connection.")

if __name__ == '__main__':
    cProfile.run('main()', 'profiling_output.stats')
