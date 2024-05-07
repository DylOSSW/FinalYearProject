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
import audioop
import pyaudio
import wave
import os
import numpy as np
from playsound import playsound
import psutil
import GPUtil

stop_event = threading.Event()

audio_input_queue = queue.Queue()
ambient_detected = False
speech_volume = 100
listening_enabled = False  # Flag to control the listening process


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
    

def play_response(audio_file_path):
    # Initialize pygame mixer
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # Check if the file is still playing
        pygame.time.Clock().tick(10)  # Wait briefly

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
        
        # Load and play the audio file using pygame
        play_response(audio_file_path)
        
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
            frames = []

        if recording:
            frames.append(data)


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

speech_processing_queue = queue.Queue() 
recognizer = sr.Recognizer()       
process_audio_thread = threading.Thread(target=process_audio_thread_function, args=(recognizer, audio_input_queue, speech_processing_queue, stop_event))

speech_thread = threading.Thread(target=live_speech_to_text, args=(speech_processing_queue,))
process_audio_thread.start()
speech_thread.start()
user_name = get_user_name()