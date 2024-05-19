import re
import os
from openai import OpenAI

# Function to clean text to ensure it's safe for filenames (optional here since we're not using it for filenames anymore)
def clean_filename(text, max_length=50):
    clean_text = re.sub(r'[\\/*?:"<>|]', "", text)
    return clean_text[:max_length]

# Function to synthesize speech and save it to a file using specified filename
def synthesize_speech_and_save(text, filename, save_dir="prompts"):
    global client
    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text
    )
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'wb') as audio_file:
        audio_file.write(response.content)
    print(f"Speech saved to {filepath}")

# Function to load OpenAI API key
def load_openai_key():
    with open('openai_api.key', 'rb') as key_file:
        key = key_file.read().decode('utf-8')
    return key

# Load the API key and initialize the OpenAI client
api_key = load_openai_key()
client = OpenAI(api_key=api_key)

# Audio files dictionary with key phrases and correct filenames
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
    "Failed to capture the user's question.": "failed_question_capture.mp3"
}

# Iterate over phrases and synthesize speech using the specified filenames
for phrase, filename in audio_files.items():
    synthesize_speech_and_save(phrase, filename)
