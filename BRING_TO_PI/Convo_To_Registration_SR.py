import time
import threading
import queue
from openai import OpenAI
import speech_recognition as sr
import os
from playsound import playsound

farewell = ["bye","goodbye","see you later"]
# Initialize OpenAI client
client = OpenAI(api_key='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')

conversation = [
    {
        "role": "system", 
        "content": "You should respond as humanly as possible. You should have a fluid \
         and natural-feeling conversation with the user as would be expected if they spoke to an actual \
         human being. Keep your answers short enough to feel like a normal conversation, don't go off on \
         a tangent when the user says something short and simple. Don't respond too short either, act interested \
         in the user, if you respond too vaguely, the conversation will seem boring. Have your own opinions,\
         if a user says something, imagine you are another human being with your own interests, respond \
         accordingly. Keep the conversation going and keep it simple yet engaging. Keep the user engaged, \
         answer any questions as needed but don't make the answers too long, unless the user specifically \
         asks for a long story. Act like you are having a conversation with someone you just met. Your answers\
         should be a maximum of 2 sentences unless specified otherwise. If a response requires more than two\
         sentences you can ask the user if they would like to hear more about the topic or not."
    }
]
# Function to interact with OpenAI for text-based conversation
def chat_with_gpt(question,name):
    global listening_enabled

    conversation.append(
        {
            "role": "user", 
            "content": f"My name is {name}, {question}"
        }
    )

    listening_enabled = False

    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=conversation)
    answer = response.choices[0].message.content.strip()
    conversation.append(
        {
            "role": "system", 
            "content": answer
        }
    )
    return answer

# Function to generate audio using OpenAI's TTS
def generate_audio(response_from_gpt,audio_file_path):
    audio_response = client.audio.speech.create(input=response_from_gpt, voice="onyx",model='tts-1')
    audio_response.stream_to_file(audio_file_path)
    

# Function to play the audio file using playsound
def play_audio(audio_file_path):
    playsound(audio_file_path)


# Function to process audio data from the queue
def process_audio_data(audio_queue, name, stop_event):

    while not stop_event.is_set():
        try:
            text = audio_queue.get()
            response_from_gpt = chat_with_gpt(text,name)
            audio_file_path = "temp_audio.mp3"  # Temporary file path for the audio
            generate_audio(response_from_gpt, audio_file_path)
            if audio_file_path is not None:
                play_audio(audio_file_path)
                #playsound(None)  # Close the audio player
                os.remove(audio_file_path)  # Remove the temporary audio file
            else:
                print("Error: Audio file path is None")

        except PermissionError as pe:
            print(f"Permission denied error: {pe}")

        except Exception as e:
            print(f"Error in audio processing: {e}")

# Function to perform live speech-to-text
def live_speech_to_text(audio_queue, stop_event):

    recognizer = sr.Recognizer()
    mic_index = 0  # Replace 0 with the correct microphone index

    with sr.Microphone(device_index=mic_index) as source:
        print("Listening for live speech...")

        while not stop_event.is_set():
            try:
                recognizer.adjust_for_ambient_noise(source)
                audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                text = recognizer.recognize_google(audio_data)
                audio_queue.put(text)
                if any(word in text.lower() for word in farewell):
                    stop_event.set()
            except sr.WaitTimeoutError:
                pass
            except Exception as e:
                print(f"Error in audio capture: {e}")

def register():
    while True:
        user = input("Press 1 for John and 2 for Bob: ")
        if user == '1':
            name = "John"
            break
        elif user == '2':
            name = "Bob"
            break
        else:
            print("Invalid input, please try again.")
    return name
def main():

    while True:
        name = register()
        audio_queue = queue.Queue()
        stop_event = threading.Event()

        # Start threads for live speech-to-text and audio processing
        speech_thread = threading.Thread(target=live_speech_to_text, args=(audio_queue, stop_event))
        speech_thread.start()
        audio_process_thread = threading.Thread(target=process_audio_data, args=(audio_queue, name, stop_event))
        audio_process_thread.start()

        # Wait for the threads to finish
        speech_thread.join()
        audio_process_thread.join()

        # Reset conversation and stop event for the next round
        conversation.clear()
        stop_event.clear()


if __name__ == '__main__':
    main()
