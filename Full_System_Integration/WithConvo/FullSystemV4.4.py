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
#from FullSystemV3_imports import live_speech_to_text, get_user_consent_for_recognition_attempt, play_audio, get_user_age, get_user_name, process_audio_data
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
import re

# Setup logging configuration
logging.basicConfig(filename='application_audit.log', level=logging.INFO, filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

modeText = "State: Idle"
# Initialize the queue once, globally.
frame_queue = queue.Queue()
recognition_frame_queue = queue.Queue()

conversation_thread = None

stop_event = threading.Event()
face_detected_event = threading.Event()             # Face Detected Event: Triggers when the system detects a face.
has_profile_event = threading.Event()               # Has Profile Event: User indicates they have a profile.
does_not_have_profile_event = threading.Event()     # No Profile Event: User indicates they do not have a profile.  
profile_completed_event = threading.Event()         # Profile Completed Event: Profile collection is completed.
recognition_success_event = threading.Event()       # Recognition Success Event: User is successfully recognized.
recognition_failure_event = threading.Event()       # Recognition Failure Event: User is not recognized.
conversation_ended_event = threading.Event()        # Conversation Ended Event: Conversation has ended.
recognition_running_event = threading.Event()

audio_input_queue = queue.Queue()
ambient_detected = False
speech_volume = 100
listening_enabled = False  # Flag to control the listening process
farewell = ["bye","goodbye","see you later"]
# Create a regex pattern that matches any of the farewell phrases, ignoring case and handling punctuation
farewell_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(word) for word in farewell) + r')\b', re.IGNORECASE)



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
    "And now, please slowly turn to your right.": "turn_right.mp3",
    "Have you previously attended this session, provided consent and registered a profile?": "previous_consent.mp3",
    
}

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
    print("Generating audio from chat's response")
    audio_response = client.audio.speech.create(input=response_from_gpt, voice="onyx",model='tts-1')
    print("Audio generated")
    audio_response.stream_to_file(audio_file_path)
    print("Audio streamed?")
    

# Function to play the audio file using playsound
def play_response(audio_file_path, retries = 3, delay = 2):
    global listening_enabled
    if listening_enabled:
        listening_enabled = False
    for attempt in range(retries):
            try:
                playsound(audio_file_path)
                break
            except Exception as e:
                print(f"Error playing audio file {audio_file_path}: {e}")
                time.sleep(delay)  # Delay before retrying
    
    #time.sleep(1)
    listening_enabled = True

def play_audio(message_key, name=None, retries = 3, delay = 2):
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
        for attempt in range(retries):
            try:
                playsound(audio_file_path)
                break
            except Exception as e:
                print(f"Error playing audio file {audio_file_path}: {e}")
                time.sleep(delay)  # Delay before retrying
        
    except Exception as e:
        print(f"Error playing audio file {audio_file_path}: {e}")
    finally:
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
            #print("Recording ENABLED")
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
                
            if farewell_pattern.search(result.text):
                if conversation_thread and conversation_thread.is_alive():
                    print("Farewell phrase registered - Setting conversation ended event")
                    conversation_ended_event.set()
            frames = []

        if recording:
            frames.append(data)


def process_audio_data(audio_input_queue):
    global listening_enabled
    print("INSIDE CONVERSATION MODE")
    print("Listening enabled: ", listening_enabled)
    if not listening_enabled:
        listening_enabled = True
        print("Listening enabled mode change: ", listening_enabled)
    while not conversation_ended_event.is_set():
        #listening_enabled = True
        try:
            text = audio_input_queue.get()
            print("Text from audio queue: ", text)
            response_from_gpt = chat_with_gpt(text)
            print("Response from chat: ", response_from_gpt)
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
    print("Getting input with retries")
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
                print(response)
                
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

def get_user_consent_for_profiling():
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
    print("name_response",name_response)
    if name_response:
        print("inside name response if")
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
    
def get_user_consent_for_recognition_attempt():
    consent_response = get_user_input_with_retries("Thank you for your consent.")
    #consent_response = get_user_input_with_retries("Have you previously attended this session, provided consent and registered a profile?")
    print("Consent response.lower(): ", consent_response.lower())
    if "yes" in consent_response.lower():
        play_audio("Thank you for your consent.")
        logging.info("User consent received.")
        return True
    else:
        play_audio("You have not given consent to process your facial features. Exiting the application.")
        logging.info("User consent has not been given.")
        return False

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


def attempt_recognition(cap, face_detection, frame_rgb, face_detected_event, conn, profile_mode_event):
    recognition_count = 0  # Variable to count successful recognitions
    retry_max = 3
    match_threshold = 3
    retry_counter = 0
    #Count positive matches
    num_matches = 0
    
    global modeText
    user_ids, embeddings = get_all_embeddings(conn)
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
    modeText = "State: Recognition"
    recognition_running_event.set()
    
    while not conversation_ended_event.is_set():
        if not recognition_frame_queue.empty():
            captured_frames = []
            matched_frame_indexes = []  # Store indexes of frames with matches
            matched_user_index = []

            captured_frames.append(recognition_frame_queue.get())

            # Capture embeddings for each frame
            captured_embeddings = []

            for frame in captured_frames:
                captured_embeddings.extend(capture_embeddings_with_mediapipe(face_detection, facenet_model, frame))
            
            if captured_embeddings:
                for index, captured_embedding in enumerate(captured_embeddings):
                    captured_embedding_np = np.array(captured_embedding).flatten()
                    closest_index = find_closest_embedding(captured_embedding_np, embeddings)
                    if closest_index != -1:
                        user_id = user_ids[closest_index]
                        print(f"Hello there, Recognized User ID {user_id}!")
                        retry_counter = 0
                        num_matches += 1
                        matched_frame_indexes.append(index)  # Store index of matching frame
                        matched_user_index.append(closest_index)
                        if num_matches >= match_threshold:
                            if conversation_thread == None:
                                print("+++Match found+++")
                                print("Matched Frame Indexes: ",matched_frame_indexes)
                                print("Matched closest user indexes: ", matched_user_index)
                                print("\n\nThree successful recognitions. Starting Conversation.")
                                #face_detected_event.set()  # Set face detected event
                                recognition_success_event.set()
                                break  # Exit the loop after successful recognition
                            else:
                                continue
                        else:
                            print(f"{3 - recognition_count} more recognitions needed for event.")

                    else:
                        retry_counter+=1
                        print(f"---Match NOT found---\nRetries left: {retry_max - retry_counter}")
                        
                        if retry_counter == retry_max:
                            retry_counter = 0
                            retry_max = 3                        

                            print("User not recognized. Switching to profiling mode.")
                            recognition_running_event.clear()
                            recognition_failure_event.set()      # Recognition Failure Event: User is not recognized.
                            
                            break  # Exit the loop after failed recognition
            recognition_frame_queue.task_done()


# Rest of the code remains the same

def start_profiling_thread(conn, cap, face_detection, frame_queue):
    global modeText
    modeText = "State: Profiling"
    # Reset the stop event in case it was set from a previous profiling session
    stop_event.clear()

    try:
        '''if not get_user_consent_for_profiling():
            print("Exiting due to lack of consent.")
            return'''
        
        user_name = get_user_name()
        user_age = get_user_age()
        #user_name = "Dylan"
        #user_age = 26
        facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
        processing_thread = threading.Thread(target=process_frames, args=(face_detection, facenet_model, conn, user_name, user_age, stop_event, frame_queue))
        processing_thread.start()

        instructions = ["Please face forward for a few seconds.", "Now, please slowly turn to your left.", "And now, please slowly turn to your right."]
        for instruction in instructions:
            play_audio(instruction)
            capture_for_duration(cap, frame_queue, duration=6)
        
        stop_event.set()
        processing_thread.join()
        print("Profiling done")

    finally:
        # Notify the main thread that profiling is done
        profile_completed_event.set()

def check_profile_state():

    global modeText
    modeText = "State: Check Profile"

    check_profile = get_user_consent_for_recognition_attempt()

    if check_profile == True:
         has_profile_event.set()
         print("User confirmed having a profile.")
    else:
         does_not_have_profile_event.set()
         print("User confirmed not having a profile.")


def main():
    mp_face_detection, stop_event, face_detected_event, profile_mode_event, speech_processing_queue = initialize_components()
    db_file = 'MYDB2.db'
    conn = create_connection(db_file)
    speech_thread = threading.Thread(target=live_speech_to_text, args=(audio_input_queue,))
    speech_thread.start()
    global modeText, listening_enabled, conversation_thread
    if conn:
        create_tables(conn)
        cap = cv2.VideoCapture(0)
        try:
            with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
                #user_ids, embeddings = get_all_embeddings(conn)
                face_detected_time = None

                while not stop_event.is_set():
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = face_detection.process(frame_rgb)
                        display_frame = frame.copy()
                        developer_frame = frame.copy()

                        if results.detections:
                            if face_detected_time is None:
                                face_detected_time = time.time()  # Start the timer on the first detection
                                                    # Example of getting operational info
                            # Mode text stays the same or can be moved to another place if it overlaps with bounding box
                            for detection in results.detections:
                                    bboxC = detection.location_data.relative_bounding_box
                                    ih, iw, _ = frame.shape
                                    bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih))
                                    cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 2)
                                    cv2.rectangle(developer_frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 2)
                                
                            cv2.putText(display_frame, modeText, (50, frame.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)


                            if time.time() - face_detected_time >= 3:  # Check if the face has been detected continuously for 3 seconds
                                if not face_detected_event.is_set():  # Check this only once
                                    face_detected_event.set()
                                    print("Face detected for 3 seconds, initiating check profile state")
                                    threading.Thread(target=check_profile_state).start()
                                    face_detected_time = None  # Reset timer after action
                                
                            
                            if recognition_running_event.is_set():
                                if recognition_frame_queue.empty():
                                    recognition_frame_queue.put(frame_rgb)

                            if does_not_have_profile_event.is_set():
                                print("Starting profiling mode.")
                                profile_thread = threading.Thread(target=start_profiling_thread, args=(conn, cap, face_detection, frame_queue))
                                profile_thread.start()
                                does_not_have_profile_event.clear()  # Reset after starting profiling

                            if has_profile_event.is_set():
                                print("Attempting Recognition.")
                                recognition_thread = threading.Thread(target=attempt_recognition, args=(cap, face_detection, frame_rgb, face_detected_event, conn, profile_mode_event))
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
                                #recognition_thread.join()
                                modeText = "State: Conversation"
                                conversation_thread = threading.Thread(target=process_audio_data, args=(audio_input_queue,))
                                conversation_thread.start()
                                recognition_success_event.clear()  # Reset after starting profiling

                            if profile_completed_event.is_set():
                                modeText = "State: Idle"
                                profile_thread.join()
                                print("Profile completed. Ready for new face detection.")
                                face_detected_time = None  # Reset to detect new face
                                face_detected_event.clear()  # Allow new face detection
                                profile_completed_event.clear()  # Reset profiling event

                            
                            if conversation_ended_event.is_set():
                                listening_enabled = False
                                modeText = "State: Idle"
                                recognition_thread.join() # Stop recognition - Assuming user leaves after exiting conversation
                                conversation_thread.join()
                                print("Conversation completed. Ready for new face detection.")
                                face_detected_time = None  # Reset to detect new face
                                face_detected_event.clear()  # Allow new face detection
                                conversation_ended_event.clear()  # Reset profiling event

                            # Drawing bounding boxes and other UI updates here...

                        else:
                            face_detected_time = None  # Reset the timer if no face is detected

                        
                      
                        ''' Operational Statistics'''

                        # Example of getting operational info
                        thread_count = threading.active_count()
                        cpu_usage = psutil.cpu_percent()
                        fps = cap.get(cv2.CAP_PROP_FPS)  # This is just an example, actual FPS calculation may vary

                        memory_usage_info = psutil.Process().memory_info()
                        ram_used = memory_usage_info.rss / (1024 ** 3)  # GB

                        gpus = GPUtil.getGPUs()
                        for gpu in gpus:
                            gpu_load = f"{gpu.load*100}%"
                            gpu_memory = f"{gpu.memoryUsed}/{gpu.memoryTotal}MB"

                        # Example of disk I/O using psutil
                        disk_io_start = psutil.disk_io_counters()
                        # ... operations that read/write to disk
                        disk_io_end = psutil.disk_io_counters()
                        read_bytes = disk_io_end.read_bytes - disk_io_start.read_bytes
                        write_bytes = disk_io_end.write_bytes - disk_io_start.write_bytes
 

                        # Initialize vertical position offset
                        vertical_pos = frame.shape[0] - 50  # Starting from bottom, going up
                        line_height = 20  # Height of each line of text

                        # Draw Threads info
                        cv2.putText(developer_frame, f"Threads: {thread_count}", 
                            (10, vertical_pos), cv2.FONT_HERSHEY_COMPLEX, 
                                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        vertical_pos -= line_height

                        # Draw CPU Usage info
                        cv2.putText(developer_frame, f"CPU: {cpu_usage}%", 
                        (10, vertical_pos), cv2.FONT_HERSHEY_COMPLEX, 
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        vertical_pos -= line_height

                        # Draw FPS info
                        cv2.putText(developer_frame, f"FPS: {fps}", 
                                (10, vertical_pos), cv2.FONT_HERSHEY_COMPLEX, 
                                0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        vertical_pos -= line_height

                        # Draw RAM Usage info
                        cv2.putText(developer_frame, f"RAM Used: {ram_used:.2f} GB", 
                            (10, vertical_pos), cv2.FONT_HERSHEY_COMPLEX, 
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        vertical_pos -= line_height

                        # Draw GPU info (assumes 'gpus' is not empty and you are displaying info for one GPU)
                        if gpus:
                            gpu = gpus[0]
                            gpu_load = f"{gpu.load * 100:.1f}%"
                            gpu_memory = f"{gpu.memoryUsed:.0f}/{gpu.memoryTotal:.0f}MB"
                            cv2.putText(developer_frame, f"GPU Load: {gpu_load}, GPU Mem: {gpu_memory}", 
                                (10, vertical_pos), cv2.FONT_HERSHEY_COMPLEX, 
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)
                            vertical_pos -= line_height

                        # Example of getting operational info
                        thread_count = threading.active_count()
                        cpu_usage = psutil.cpu_percent()
                        fps = cap.get(cv2.CAP_PROP_FPS)  # This is just an example, actual FPS calculation may vary

                        # Create the operational info text
                        operational_info = f"Threads: {thread_count}, CPU: {cpu_usage}%, FPS: {fps}"

                        # Position for the operational info text
                        operational_info_x = 10  # Some padding from the left edge of the window
                        operational_info_y = frame.shape[0] - 30  # Some padding from the bottom'''

                        ''' Operational Statistics'''

                        # Draw the operational info text on the frame
                        cv2.putText(developer_frame, operational_info, (operational_info_x, operational_info_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                        cv2.imshow('User View', display_frame)
                        cv2.imshow('Developer View', developer_frame)



        
        
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
