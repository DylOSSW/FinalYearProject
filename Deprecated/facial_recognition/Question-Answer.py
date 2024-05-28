import time
from datetime import datetime
import speech_recognition as sr
import pyttsx3
import threading
import queue
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key='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')

# Function to convert text to speech
def text_to_speech(text, language='en'):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Function to interact with OpenAI for text-based conversation
def chat_with_gpt(question):
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ]
    preliminary_response = "Hang on a minute, I'm thinking about your question."
    text_to_speech(preliminary_response)
    time.sleep(2)
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=conversation)
    answer = response.choices[0].message.content.strip()
    return answer

# Function to perform live speech-to-text
def live_speech_to_text(audio_queue):
    recognizer = sr.Recognizer()
    mic_index = 0  # Replace 0 with the correct microphone index

    with sr.Microphone(device_index=mic_index) as source:
        print("Listening for live speech...")

        while True:
            try:
                recognizer.adjust_for_ambient_noise(source)
                audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                audio_queue.put(audio_data)
            except sr.WaitTimeoutError:
                pass
            except Exception as e:
                print(f"Error in audio capture: {e}")

# Function to process audio data from the queue
def process_audio_data(audio_queue):
    recognizer = sr.Recognizer()

    while True:
        try:
            audio_data = audio_queue.get()
            transcript = recognizer.recognize_google(audio_data)
            print("Transcript:", transcript)
            response_from_gpt = chat_with_gpt(transcript)
            text_to_speech(response_from_gpt)

        except Exception as e:
            print(f"Error in audio processing: {e}")

def main():
    audio_queue = queue.Queue()

    # Start threads for live speech-to-text and audio processing
    speech_thread = threading.Thread(target=live_speech_to_text, args=(audio_queue,))
    speech_thread.start()
    audio_process_thread = threading.Thread(target=process_audio_data, args=(audio_queue,))
    audio_process_thread.start()

    try:
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == '__main__':
    main()
