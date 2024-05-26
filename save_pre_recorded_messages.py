# Student Names:   Dylan Holmwood and Kristers Martukans
# Student Numbers: D21124331 and D21124318
# Date:            29th May 2024
# Module Title:    Final Year Project
# Module Code:     PROJ4004
# Supervisors:     Paula Kelly and Damon Berry
# Script Name:     save_pre_recorded_messages.py
# Description:     This script synthesizes speech for given text phrases using the OpenAI API and saves the resulting audio files
#                  with specified filenames. The script loads the OpenAI API key, iterates over a dictionary of text phrases and 
#                  filenames, and saves the synthesized speech audio files to a specified directory.

""" Libraries """
import os
from openai import OpenAI
from audio_files import audio_files

# Function to load OpenAI API key
def load_openai_key():
    """Load the OpenAI API key from a file."""
    with open('openai_api.key', 'rb') as key_file:
        key = key_file.read().decode('utf-8')  # Read and decode the API key
    return key

# Function to synthesize speech and save it to a file using specified filename
def synthesize_speech_and_save(text, filename, save_dir="prompts"):
    """ Synthesize speech from the given text and save it to a file. """
    
    api_key = load_openai_key()  # Load the OpenAI API key
    client = OpenAI(api_key=api_key)  # Initialize the OpenAI client with the API key

    # Synthesize speech using the OpenAI API
    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text
    )

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)  # Define the path to save the audio file

    # Write the synthesized speech to a file
    with open(file_path, 'wb') as audio_file:
        audio_file.write(response.content)
    
    print(f"Speech saved to {file_path}")  # Print the path where the audio file is saved

# Iterate over phrases and synthesize speech using the specified filenames
for phrase, fname in audio_files.items():
    synthesize_speech_and_save(phrase, fname)  # Call the function to synthesize and save speech for each phrase
