from pathlib import Path
from openai import OpenAI
import pygame
import speech_recognition as sr

# Initialize the OpenAI client with your API key
client = OpenAI(api_key='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')

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak your question:")
        audio = recognizer.listen(source)
    try:
        question = recognizer.recognize_google(audio)
        print("You asked:", question)
        return question
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand what you said.")
        return None
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return None

# Function to chat with GPT
def chat_with_gpt(question):
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ]
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=conversation)
    answer = response.choices[0].message.content.strip()
    return answer

# Function to synthesize speech
def synthesize_speech(text):
    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text
    )
    audio_file_path = "output.mp3"
    with open(audio_file_path, 'wb') as audio_file:
        audio_file.write(response.content)
    return audio_file_path

# Initialize pygame mixer
pygame.mixer.init()

# Main function
def main():
    while True:
        question = recognize_speech()
        if question:
            answer = chat_with_gpt(question)
            audio_file_path = synthesize_speech(answer)
            pygame.mixer.music.load(audio_file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

if __name__ == "__main__":
    main()
