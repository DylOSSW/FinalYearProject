# Student Names:   Dylan Holmwood and Kristers Martukans
# Student Numbers: D21124331 and D21124318
# Date:            21st May 2024
# Module Title:    Final Year Project
# Module Code:     PROJ4004
# Supervisors:     Paula Kelly and Damon Berry 
# Script Name:     MainSystem.py
# Description:     The script is a Python-based facial recognition and voice command system designed 
#                  for interactive user engagement. It includes functionalities such as voice-to-text conversion, user recognition 
#                  through facial features, and audio feedback through text-to-speech. The application can create and retrieve user profiles, 
#                  manage sessions based on user interactions, and maintains logs for auditing purposes. It utilizes threading to handle multiple operations 
#                  simultaneously, such as processing speech and detecting faces, ensuring efficient management of tasks.

# Import necessary modules and functions
from import_libraries import *
from audio_files import audio_files
from conversation_setup import conversation_initial_setup
from operational_stats_show import display_operational_stats
from database_operations import (
    create_connection, create_tables, insert_user_profile, get_all_embeddings,
    get_returning_user_name, insert_embeddings, delete_old_records, delete_database_on_exit
)

# Setup logging configuration for application audit
logging.basicConfig(
    filename='application_audit.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Define system states and initialization
modeText = "State: Idle"                        # Initial state of the application
display_info_default = {"Name": "Unknown"}       # Default display info
display_info = display_info_default.copy()

# Initialize queues for managing data flow
frame_queue = queue.Queue()                     # Queue for raw frames from camera
recognition_frame_queue = queue.Queue()         # Queue for frames to be processed for recognition
display_info_queue = queue.Queue()

# Define threading events for process control
stop_event = threading.Event()                  # Event to signal stopping all threads
face_detected_event = threading.Event()         # Triggers when a face is detected
has_profile_event = threading.Event()           # Indicates the user has a profile
does_not_have_profile_event = threading.Event() # Indicates the user does not have a profile
profile_completed_event = threading.Event()     # Signals that profile collection is completed
recognition_success_event = threading.Event()   # User is successfully recognized
recognition_failure_event = threading.Event()   # User is not recognized
conversation_ended_event = threading.Event()    # Conversation has ended
recognition_running_event = threading.Event()   # Recognition process is active
returning_user_event = threading.Event()        # Existing user returned
profile_created = threading.Event()             # New profile has been created

# Control flags for various operations
conversation_running = False                    # State flag for conversation process
profiling_running = False                       # State flag for profiling process
recognition_running = False                     # State flag for recognition process

# Queue for audio input and related variables
audio_input_queue = queue.Queue()               # Queue for audio input data
ambient_detected = False                        # Flag to indicate ambient noise detection
speech_volume = 100                             # Control for speech volume
listening_enabled = False                       # Flag to control the listening process

# User information variables
user_name = "Unknown"                           # Default user name
user_age = "Unknown"                            # Default user age

# Copy the initial setup for conversation from a predefined setup
conversation = conversation_initial_setup.copy()

# Placeholder for threads (if any) for different operations
conversation_thread = None
recognition_thread = None

def load_openai_key():
    with open('openai_api.key', 'rb') as key_file:
        key = key_file.read().decode('utf-8')  # Decode bytes to string
    return key

api_key = load_openai_key()

# Initialize the OpenAI client with the api_key as a named (keyword) argument
client = OpenAI(api_key=api_key)

def chat_with_gpt(question):
    # Append the user's question to the conversation
    conversation.append({"role": "user", "content": f"{question}"})

    # Make a request to the OpenAI API to generate a response
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=conversation)

    # Extract the generated response from the API response
    answer = response.choices[0].message.content.strip()

    # Append the generated answer to the conversation
    conversation.append({"role": "assistant", "content": answer})

    # Return the generated answer
    return answer

# Function to print conversation
def print_conversation(conversation):
    for message in conversation:
        role = message["role"]
        content = message["content"]
        print(f"{role.capitalize()}: {content}")

# Function to generate audio using OpenAI's TTS
def generate_audio(response_from_gpt,audio_file_path):
    print("Generating audio from chat's response")
    audio_response = client.audio.speech.create(input=response_from_gpt, voice="onyx",model='tts-1')
    print("Audio generated")
    audio_response.stream_to_file(audio_file_path)
    print("Audio streamed?")
    
# Function to play the audio file using playsound
def play_response(audio_file_path, retries=3, delay=2):
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
    #listening_enabled = True

def play_audio(message_key, name=None, retries=3, delay=2):

    global listening_enabled
    audio_file_path = None  # Initialize outside the try block for broader scope

    try:
        # Temporarily disable listening to prevent echo or feedback during playback
        if listening_enabled:
            listening_enabled = False

        # Format the message key with the name if necessary
        if name and '{name}' in message_key:
            message_key = message_key.format(name=name)

        # Check if the message key exists in the audio files dictionary
        if message_key not in audio_files:
            raise KeyError(f"Audio file key not found: {message_key}")

        # Construct the path to the audio file
        audio_file_path = 'prompts/' + audio_files[message_key]
        print(f"Playing audio: {audio_file_path}")

        # Attempt to play the audio file, retry on failure
        for attempt in range(retries):
            try:
                playsound(audio_file_path)
                break  # Exit loop if successful
            except Exception as e:
                print(f"Error playing audio file {audio_file_path}: {e}")
                time.sleep(delay)  # Wait before retrying

    except KeyError as e:
        print(e)  # Log the error for diagnostics
        raise  # Re-raise to notify the calling function of the failure
    finally:
        # Re-enable listening after audio playback or on error
        listening_enabled = True

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y
   
def live_speech_to_text(audio_input_queue, wait_time=70):
    global ambient_detected
    global speech_volume
    global listening_enabled

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    frames = []
    recording = False
    frames_recorded = 0

    while True:
        frames_recorded += 1
        data = stream.read(CHUNK, exception_on_overflow=False)  # Read data from the stream

        # Convert byte data to numpy array
        audio_data = np.frombuffer(data, dtype=np.int16)
        # Filter the data
        filtered_data = bandpass_filter(audio_data, 300, 3400, RATE)

        # Convert back to bytes
        data = filtered_data.astype(np.int16).tobytes()

        # Calculate RMS of the filtered data
        rms = audioop.rms(data, 2)

        if not ambient_detected:
            if frames_recorded < 40:
                if frames_recorded == 1:
                    print("Detecting ambient noise...")
                if speech_volume < rms:
                    speech_volume = rms
                continue
            elif frames_recorded == 40:
                print("Listening...")
                speech_volume = speech_volume * 3  # Consider adjusting this based on testing
                ambient_detected = True

        if rms > speech_volume and listening_enabled:
            recording = True
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
            os.remove("audio.wav")  # Remove the audio file

            print("Result org: ", result)
            print("Result text: ", result.text)
            audio_input_queue.put(result.text)
            
            farewell = ["bye","goodbye","see you later"]
            # Create a regex pattern that matches any of the farewell phrases, ignoring case and handling punctuation
            farewell_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(word) for word in farewell) + r')\b', re.IGNORECASE)
            
            if farewell_pattern.search(result.text):
                if conversation_thread and conversation_thread.is_alive():
                    print("Farewell phrase registered - Setting conversation ended event")
                    conversation_ended_event.set()
                
            frames = []

        if recording:
            frames.append(data)

def process_audio_data(audio_input_queue):
    """
    Args:
    audio_input_queue (queue.Queue): A queue containing audio data inputs.
    """
    global listening_enabled, conversation_running  # Referencing global variables for state management

    print("INSIDE CONVERSATION MODE")
    print("Listening enabled: ", listening_enabled)
    
    # Ensure listening is enabled at the start of the conversation
    if not listening_enabled:
        listening_enabled = True
        print("Listening enabled mode change: ", listening_enabled)

    # Continue processing until a termination event is set
    while not conversation_ended_event.is_set() and not recognition_failure_event.is_set():
        try:
            # Retrieve text from the audio input queue
            text = audio_input_queue.get()
            if text is None:  # Check if the text is None to exit loop
                print("Exiting Conversation Loop")
                clear_queue(audio_input_queue)
                break
            print("Text from audio queue: ", text)
            
            # Generate a response using a chat function with GPT
            response_from_gpt = chat_with_gpt(text)
            print("Response from chat: ", response_from_gpt)
            
            audio_file_path = "temp_audio.mp3"  # Temporary file path for storing the audio response
            generate_audio(response_from_gpt, audio_file_path)  # Generate audio from the response
            
            # Play the generated audio response and clean up the file
            if audio_file_path:
                play_response(audio_file_path)
                os.remove(audio_file_path)  # Remove the temporary audio file after playing
            else:
                print("Error: Audio file path is None")
            listening_enabled = True  # Ensure listening is enabled after processing

        except queue.Empty:
            continue  # Handle empty queue exceptions and continue the loop
        except PermissionError as pe:
            print(f"Permission denied error: {pe}")  # Log permission errors explicitly
        except Exception as e:
            print(f"Error in audio processing: {e}")  # General exception handling for unexpected errors
        finally:
            clear_queue(audio_input_queue)  # Clear the queue at the end of processing
            conversation_running = False  # Update the state to indicate the conversation has ended

def get_user_name(prompt_key="Please say your name.", attempt_limit=3, timeout=10):
    global listening_enabled
    attempts = 0
    while attempts < attempt_limit:
        if not listening_enabled:
            listening_enabled = True

        start_time = time.time()  # Record the start time for each attempt
        try:
            # Play the prompt from a pre-recorded file
            play_audio(prompt_key)
            while time.time() - start_time < timeout:  # Check elapsed time
                try:
                    response = audio_input_queue.get(timeout=timeout)
                    print("Received:", response)

                    # Analyze the response to determine if it's clear
                    prompt_analysis = (
                        f"Please analyze the response: '{response}'. I want you to only return the name "
                         "as a single word or the word 'unclear'. If there is a reasonable name, just return "
                         "that; if there is an unclear sentence where you can't make out the name, just return "
                         "'unclear'. Only return the name or the word unclear. A one word single response nothing else."
                    )

                    name = chat_with_gpt(prompt_analysis).strip().lower()
                    print("for debuggin", name)
                    if "unclear" in name:
                        print("The response was unclear or invalid.")
                        play_audio("I couldn't understand the name, please try again.")
                        attempts += 1  # Ensure the next attempt increments if the response is unclear
                        break  # Exit this attempt and try again
                    else:
                        print("Chat Answer: ", name)
                        logging.info("User name has been received: %s", name)
                        play_audio("Thank you for providing your name")
                        listening_enabled = False
                        return name  # Return the clear name

                except queue.Empty:
                    play_audio("I didn't catch that. Let's try again.")

        except Exception as e:
            print("Error while processing input:", str(e))
            play_audio("An error occurred. Let's try again.")

        # If timeout is reached without a valid response
        if time.time() - start_time >= timeout:
            play_audio("Sorry, I couldn't hear anything.")
            attempts += 1

    listening_enabled = False
    return None  # Return None if all attempts fail

def get_user_age(prompt_key="Please tell me your age.", attempt_limit=3, timeout=10):
    global listening_enabled
    attempts = 0
    while attempts < attempt_limit:
        if not listening_enabled:
            listening_enabled = True

        start_time = time.time()  # Record the start time for each attempt
        try:
            # Play the prompt from a pre-recorded file
            play_audio(prompt_key)
            while time.time() - start_time < timeout:  # Check elapsed time
                try:
                    response = audio_input_queue.get(timeout=timeout)
                    print("Received:", response)

                    # Analyze the response to determine if it's clear and valid
                    prompt_analysis = (
                        f"Please analyze the response: '{response}'. I want you to only return the age "
                         "or the word 'unclear'. If there is a reasonable age, just return that; if there is "
                         "an unclear sentence where you can't make out the age, just return 'unclear'. "
                         "If you return the age, do not include a full stop or any other words, just the age. "
                         "Whether it be age or unclear, you should only return a one word response."
                        )

                  
                    age_text = chat_with_gpt(prompt_analysis).strip().lower()
                    print("Debugging:", age_text)
                    if "unclear" in age_text:
                        print("The response was unclear or invalid.")
                        play_audio("Sorry I couldn't make out your age, could you please say it again?.")
                        attempts += 1  # Ensure the next attempt increments if the response is unclear
                        break  # Exit this attempt and try again
                    else:
                        try:
                            age = int(age_text)
                            print("Extracted Age:", age)
                            logging.info("User age has been received: %d", age)
                            play_audio("Thank you. I have recorded your age.")
                            listening_enabled = False
                            return age  # Return the age if it's clear and valid
                        except ValueError:
                            print("Failed to extract a valid age.")
                            play_audio("Your response didn't seem to have a clear age in it.")
                            attempts += 1
                            break

                except queue.Empty:
                    play_audio("I didn't catch that. Let's try again.")

        except Exception as e:
            print("Error while processing input:", str(e))
            play_audio("An error occurred. Let's try again.")

        # If timeout is reached without a valid response
        if time.time() - start_time >= timeout:
            play_audio("Sorry, I couldn't hear anything.")
            attempts += 1

    listening_enabled = False
    return None  # Return None if all attempts fail

def get_user_input_with_retries(prompt_key, attempt_limit=3, timeout=10):
    global listening_enabled
    attempts = 0
    
    while attempts < attempt_limit:
        if not listening_enabled:
            listening_enabled = True
        
        # Play the initial prompt
        play_audio(prompt_key)
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                remaining_time = timeout - (time.time() - start_time)
                response = audio_input_queue.get(timeout=remaining_time)
                return response
            except queue.Empty:
                # If there's still time left, continue waiting for response
                if time.time() - start_time >= timeout:
                    break
        
        attempts += 1
        
        if attempts < attempt_limit:
            if attempts == 2:
                play_audio("I couldn't understand that. Let's try again.")
            else:
                play_audio("I didn't catch that. Let's try again.")
        else:
            play_audio("Sorry, I couldn't hear anything.")
    
    listening_enabled = False
    return None

def check_profile_state():
    clear_queue(audio_input_queue)
    global modeText
    modeText = "State: Check Profile"
    logging.info("Check Profile State")
    
    play_audio("Hello there my name is Onyx.")
    check_profile = get_user_consent_for_recognition_attempt()

    if check_profile:
        has_profile_event.set()
        logging.info("User confirmed having a profile.")
        print("User confirmed having a profile.")
    else:
        does_not_have_profile_event.set()
        logging.info("User confirmed not having a profile.")
        print("User confirmed not having a profile.")

def get_user_consent_for_profiling():
    # Brief explanation of the session's purpose
    brief_explanation =  "During this session, we'll capture and analyze your facial features and demographic data to personalize your experience. Would you like to listen to our full data handling statement?"
    consent_response = get_user_input_with_retries(brief_explanation)
    
    # Provide more details to the user
    if consent_response is not None and "yes" in consent_response.lower():
        play_audio("This statement outlines the data privacy and security protocols employed during this session. "
                   "Facial features are collected for the purpose of facial recognition, and demographic information "
                   "such as name and age are used to personalize the user experience. All data collected is encrypted "
                   "and securely stored. Importantly, this data is retained for a maximum duration of one hour before "
                   "being automatically deleted, thereby ensuring the privacy and security of user information. "
                   "Furthermore, it's important to understand that during this session, your voice data will be sent to OpenAI, "
                   "a third-party provider, for speech-to-text (speech recognition) conversion. OpenAI processes this data solely "
                   "for transcription purposes and returns the transcription result, adhering to a strict zero data retention policy "
                   "with its Whisper transcription service. This ensures that your privacy is upheld and no voice inputs are stored.")

    consent_response = get_user_input_with_retries("Do you consent to have your facial features captured and analyzed for this session? Please say 'yes' or 'no'.")
    print("Consent response.lower(): ", consent_response.lower())
    if consent_response is not None and "yes" in consent_response.lower():
        play_audio("Thank you for your consent.")
        logging.info("User consent received.")
        return True
    else:
        play_audio("You have not given consent to process your facial features. Exiting the application.")
        logging.info("User consent has not been given.")
        return False
        

def get_user_consent_for_recognition_attempt():
    consent_response = get_user_input_with_retries("Have you previously attended this session, provided consent and registered a profile?")
    
    if consent_response is not None and "yes" in consent_response.lower():
        play_audio("Thank you for your consent.")
        logging.info("User consent received.")
        return True
    else:
        play_audio("Oh, you don't have a profile? Let's get one setup for you!")
        logging.info("User does not have a profile.")
        return False

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

def capture_for_duration(duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        time.sleep(0.01)  # Makes the loop wait for 10ms
def find_closest_embedding(captured_embedding, embeddings, threshold=0.6):
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
        logging.info("No matching embedding found.")
        return -1

    return closest_embedding_index

def attempt_recognition(face_detection, conn):
    recognition_count = 0  # Variable to count successful recognitions
    retry_max = 10
    match_threshold = 5
    retry_counter = 0
    #Count positive matches
    num_matches = 0
    print("Matches: ",num_matches)
    print(recognition_count, retry_max, match_threshold, retry_counter)
    global modeText
    user_ids, embeddings = get_all_embeddings(conn)
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
    modeText = "State: Recognition"
    logging.info("Recognition State")
    recognition_running_event.set()
    
    while not recognition_failure_event.is_set():
        if conversation_ended_event.is_set():
            break
        if not recognition_frame_queue.empty():
            captured_frames = []
            matched_frame_indexes = []  # Store indexes of frames with matches
            matched_user_index = []

            frame = recognition_frame_queue.get()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            captured_frames.append(frame_rgb)

            # Capture embeddings for each frame
            captured_embeddings = []

            try:
                for frame in captured_frames:
                    captured_embeddings.extend(capture_embeddings_with_mediapipe(face_detection, facenet_model, frame))
                
                if captured_embeddings:
                    for index, captured_embedding in enumerate(captured_embeddings):
                        captured_embedding_np = np.array(captured_embedding).flatten()
                        closest_index = find_closest_embedding(captured_embedding_np, embeddings)
                        if closest_index != -1:
                            user_id = user_ids[closest_index]
                            #print(f"Hello there, Recognized User ID {user_id}!")
                            retry_counter = 0
                            num_matches += 1
                            matched_frame_indexes.append(index)  # Store index of matching frame
                            matched_user_index.append(closest_index)
                            if num_matches >= match_threshold:
                                num_matches = 0
                                if returning_user_event.is_set():
                                    existing_user_name = get_returning_user_name(conn, user_id)
                                    display_info_queue.put(existing_user_name)
                                    returning_user_remark = f"It's me, {existing_user_name}."
                                    print(returning_user_remark)
                                    returning_user_greeting = chat_with_gpt(returning_user_remark)
                                    audio_file_path = "temp_audio.mp3"  # Temporary file path for the audio
                                    generate_audio(returning_user_greeting, audio_file_path)
                                    
                                    if audio_file_path is not None:
                                        play_response(audio_file_path)
                                        os.remove(audio_file_path)  # Remove the temporary audio file
                                    else:
                                        print("Error: Audio file path is None")
                                    
                                    returning_user_event.clear()
                                if conversation_thread == None:
                                    print("+++Match found+++")
                                    print("Matched Frame Indexes: ",matched_frame_indexes)
                                    print("Matched closest user indexes: ", matched_user_index)
                                    print("\n\nThree successful recognitions. Starting Conversation.")
                                    recognition_success_event.set()
                                    break  # Exit the loop after successful recognition
                                else:
                                    continue
                            else:
                                print(f"{match_threshold - num_matches} more recognitions needed for event.")

                        else:
                            num_matches = 0
                            retry_counter+=1
                            print(f"---Match NOT found---\nRetries left: {retry_max - retry_counter}")
                            
                            if retry_counter == retry_max:
                                retry_counter = 0
                                #retry_max = 10                      

                                print("User not recognized. Switching to profiling mode.")
                                display_info_queue.put("Unknown")
                                recognition_running_event.clear()
                                recognition_failure_event.set()      # Recognition Failure Event: User is not recognized.
                                
                                break  # Exit the loop after failed recognition
            except Exception as e:
                logging.error(f"Error processing frame for recognition: {e}")
            finally:
                recognition_frame_queue.task_done()

# Processes frames from the frame queue to extract and store facial embeddings and user profile     
def process_frames(face_detection, facenet_model, conn, user_name, user_age, frame_queue):

    
    # Insert user profile into the database
    user_id = insert_user_profile(conn, user_name, user_age)
    logging.info("Started processing frames.")

    # Process each frame in the queue as long as the profile creation event is active
    while profile_created.is_set():
        if not frame_queue.empty():
            image = frame_queue.get()  # Retrieve the next frame from the queue
            if image is None or image.size == 0:
                frame_queue.task_done()  # Mark the frame as processed if it is empty or invalid
                continue

            # Convert the image to RGB color space for processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
                # Capture facial embeddings using the provided models
                embeddings = capture_embeddings_with_mediapipe(face_detection, facenet_model, image_rgb)
                for embedding in embeddings:
                    # Convert the embedding to a NumPy array if it's not already
                    numpy_embedding = embedding.cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding
                    # Insert the embedding into the database for the current user
                    insert_embeddings(conn, user_id, numpy_embedding.flatten())
            except Exception as e:
                logging.error(f"Error processing frame: {e}") 
            finally:
                frame_queue.task_done()  # Mark the frame as processed in the queue

    logging.info("Stopped processing frames.")  


# Initiates user profiling in a separate thread after verifying consents and gathering necessary user details.
def start_profiling_thread(conn, face_detection, frame_queue):
    global modeText
    modeText = "State: Profiling"  # Update the application's state to 'Profiling'
    logging.info("Profile State")
    stop_event.clear()  # Reset the stop event from any previous sessions

    try:
        # Check for user consent before profiling
        if not get_user_consent_for_profiling():
            logging.info("Exiting Profiling due to lack of consent.")
            return  # Exit the function if consent is not given
        
        play_audio("Okay, so I will now begin to create a profile for you.")
        
        # Retrieve user name and age
        user_name = get_user_name()
        if not user_name:  # Check if user name was not obtained
            logging.info("User name not obtained, exiting profiling.")
            play_audio("Sorry I could not make out your name, I am going to exit profiling.") 
            return  # Exit if user name is not retrieved
        
        user_age = get_user_age()
        if not user_age:  # Check if user age was not obtained
            logging.info("User age not obtained, exiting profiling.")
            play_audio("Sorry I could not make out your age, I am going to exit profiling.")
            return  # Exit if user age is not retrieved

        # Load the facial recognition model
        logging.info("Loading facenet model.")
        facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
        logging.info("Facenet model loaded successfully.")

        # Start a thread to process frames and capture facial data
        logging.info("Starting processing thread.")
        processing_thread = threading.Thread(target=process_frames, args=(face_detection, facenet_model, conn, user_name, user_age, frame_queue))
        processing_thread.start()
        
        profile_created.set()  # Signal that the profile creation process has started
        logging.info("Profile created event set.")
        logging.info("Giving user instructions")
        
        # Provide instructions for capturing facial features from multiple angles
        instructions = ["I will now peform some calibration to capture your facial features from a few angles for better accuracy.",
                        "Please listen to the following instructions.",
                        "Please face forward for a few seconds.", 
                        "Now, please slowly turn to your left.", 
                        "And now, please slowly turn to your right."]
        
        for instruction in instructions:
            play_audio(instruction)
            capture_for_duration(duration=6)  # Capture data for a fixed duration

        profile_created.clear()  # Clear the profile creation signal
        logging.info("Profile created event cleared.")
        play_audio("Thank you, for providing your information, your profile is now complete!")
        
        processing_thread.join()  # Wait for the processing thread to finish
        logging.info("Profiling done")
        print("profiling done")

    finally:
        profile_completed_event.set()  # Notify that profiling is completed
        logging.info("Profiling completed event set.")

# Checks the user's profile state by asking if they have previously registered.
def check_profile_state():
    global modeText
    modeText = "State: Check Profile"  
    logging.info("Check Profile State")  

    # Prompt user for confirmation if they have previously registered a profile
    check_profile = get_user_consent_for_recognition_attempt()

    if check_profile:
        has_profile_event.set()  # Set the event signaling that the user has a profile
        logging.info("User confirmed having a profile.") 
        print("User confirmed having a profile.") 
    else:
        does_not_have_profile_event.set()  # Set the event indicating no existing profile
        logging.info("User confirmed not having a profile.")  
        print("User confirmed not having a profile.")  


def clear_queue(queue):
    with queue.mutex:
        queue.queue.clear()

def main():
    start_time = time.time()
    mp_face_detection = mp.solutions.face_detection
    db_file = 'MYDB2.db'
    conn = create_connection(db_file)
    speech_thread = threading.Thread(target=live_speech_to_text, args=(audio_input_queue,))
    # Start a thread to periodically delete old records
    #deletion_thread = threading.Thread(target=delete_old_records, args=(conn,))
    #deletion_thread.daemon = True  # Set the thread as a daemon thread
    #deletion_thread.start()
    speech_thread.start()
    global modeText, listening_enabled, conversation_thread, recognition_thread, conversation, conversation_initial_setup, conversation_running, display_info, display_info_default
    no_detection_counter = 0 # Number of consecutive failed detections
    NO_DETECTION_THRESHOLD = 10  # Number of consecutive frames with no detection before taking action

    if conn:
        create_tables(conn)
        cap = cv2.VideoCapture(0)
        try:
            with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
                face_detected_time = None

                while not stop_event.is_set():
                    ret, frame = cap.read()
                    #if ret:
                    if not ret or frame is None or frame.size == 0:
                        continue
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(frame_rgb)
                    display_frame = frame.copy()
                    #developer_frame = frame.copy()

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
                                #cv2.rectangle(developer_frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 2)
                            
                        cv2.putText(display_frame, modeText, (50, frame.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                        if not display_info_queue.empty():
                            user_name = display_info_queue.get()
                            display_info["Name"] = user_name
                        display_text = f"Name: {display_info['Name']}"
                        cv2.putText(display_frame, display_text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            

                        if time.time() - face_detected_time >= 3:  # Check if the face has been detected continuously for 3 seconds
                            if not face_detected_event.is_set():  # Check this only once
                                display_info = display_info_default
                                face_detected_event.set()
                                print("Face detected for 3 seconds, initiating check profile state")
                                check_profile_thread = threading.Thread(target=check_profile_state)
                                check_profile_thread.start()
                                face_detected_time = None  # Reset timer after action
                            
                        
                        if recognition_running_event.is_set():
                            if recognition_frame_queue.empty():
                                recognition_frame_queue.put(frame_rgb)

                        if does_not_have_profile_event.is_set():
                            print("Starting profiling mode.")
                            clear_queue(frame_queue)
                            profile_thread = threading.Thread(target=start_profiling_thread, args=(conn, face_detection, frame_queue))
                            profile_thread.start()
                            check_profile_thread.join()
                            does_not_have_profile_event.clear()  # Reset after starting profiling

                        if has_profile_event.is_set():
                            returning_user_event.set()
                            clear_queue(recognition_frame_queue)

                            print("Attempting Recognition.")
                            recognition_thread = threading.Thread(target=attempt_recognition, args=(face_detection, conn))
                            recognition_thread.start()
                            check_profile_thread.join()
                            has_profile_event.clear()  # Reset after starting profiling

                        if recognition_failure_event.is_set():
                            listening_enabled = False
                            print("---RECOGNITION FAILURE EVENT IS SET---")
                            recognition_thread.join()
                            print("---RECOG THREAD JOINED---")
                            if conversation_thread and conversation_thread.is_alive():
                                audio_input_queue.put(None)
                                print("---CONVO ALSO ON : ENDING---")
                                conversation_thread.join()
                                conversation_thread = None
                                conversation = conversation_initial_setup.copy()
                            clear_queue(audio_input_queue)
                            face_detected_time = None  # Reset to detect new face
                            face_detected_event.clear()  # Allow new face detection
                            recognition_failure_event.clear()  # Reset after starting profiling

                        if recognition_success_event.is_set():
                            if not conversation_running:
                                conversation_running = True
                            print("Recognition Successful.")
                            clear_queue(audio_input_queue)
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
                            print("---CONVO ENDED EVENT SET---")
                            clear_queue(audio_input_queue)
                            listening_enabled = False
                            modeText = "State: Idle"
                            conversation_thread.join()
                            conversation_thread = None
                            if recognition_thread and recognition_thread.is_alive():
                                #recognition_failure_event.set()
                                recognition_thread.join() # Stop recognition - Assuming user leaves after exiting conversation
                            conversation = conversation_initial_setup.copy()
                            print("Conversation completed and cleared. Ready for new face detection. \
                                    Conversation contents:\n")
                            print_conversation(conversation)
                            clear_queue(audio_input_queue)
                            face_detected_time = None  # Reset to detect new face
                            face_detected_event.clear()  # Allow new face detection
                            conversation_ended_event.clear()  # Reset profiling event

                        
                        if profile_created.is_set():
                            frame_queue.put(frame_rgb)  # Correct use of queue instance

                        # Drawing bounding boxes and other UI updates here...

                    else:
                        no_detection_counter += 1
                        print(f"No detection counter: {no_detection_counter}")
                        if no_detection_counter >= NO_DETECTION_THRESHOLD:
                            print("No detection threshold reached - Resetting")
                            face_detected_time = None  # Reset the timer if no face is detected
                            display_info = display_info_default
                            if recognition_thread and recognition_thread.is_alive():
                                recognition_failure_event.set()
                                print("recog alive face not detected")
                            no_detection_counter = 0

                    cv2.imshow('User View', display_frame)
                    display_operational_stats(cap, frame, start_time)
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

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats("profile_result.prof")
