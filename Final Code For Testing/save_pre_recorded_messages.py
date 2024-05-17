"""
This module provides functions to synthesize speech from text using the OpenAI
Text-to-Speech (TTS) API and save the audio files. It includes functionality to
load the API key, clean text for filenames, and map key phrases to audio file names.
"""
import os
from openai import OpenAI
from audio_files import audio_files

# Function to load OpenAI API key
def load_openai_key():
    """Load the OpenAI API key from a file."""
    with open('openai_api.key', 'rb') as key_file:
        key = key_file.read().decode('utf-8')
    return key

# Function to synthesize speech and save it to a file using specified filename
def synthesize_speech_and_save(text, filename, save_dir="prompts"):
    """
    Synthesize speech from the given text and save it to a file.

    """
    api_key = load_openai_key()
    client = OpenAI(api_key=api_key)

    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text
    )

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)
    with open(file_path, 'wb') as audio_file:
        audio_file.write(response.content)
    print(f"Speech saved to {file_path}")

# Iterate over phrases and synthesize speech using the specified filenames
for phrase, fname in audio_files.items():
    synthesize_speech_and_save(phrase, fname)
