import time
import threading
import queue
from openai import OpenAI
import os
from playsound import playsound
import audioop
import pyaudio
import wave
import os

def initialize_audio():
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
    return audio, stream, CHUNK

def detect_ambient_noise(stream, CHUNK, initial_frames=100):
    print("Detecting ambient noise...")
    total_rms = 0
    for _ in range(initial_frames):
        data = stream.read(CHUNK)
        rms = audioop.rms(data, 2)
        total_rms += rms
    average_rms = total_rms / initial_frames
    print(f"Ambient noise level detected at: {average_rms}")
    return average_rms * 1.5  # Increase threshold by 50%

def live_speech_to_text(audio_queue, client):
    audio, stream, CHUNK = initialize_audio()
    speech_volume = detect_ambient_noise(stream, CHUNK)
    listening_enabled = True
    buffer_frames = []
    recording = False
    buffer_length = 20  # Number of frames to buffer before making a start decision
    frames = []
    frames_since_noise = 0  # Counter to keep track of quiet frames

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            rms = audioop.rms(data, 2)

            # Buffer incoming audio
            buffer_frames.append((data, rms))
            if len(buffer_frames) > buffer_length:
                buffer_frames.pop(0)

            # Check if average volume in buffer is above the threshold
            average_rms = sum(frame[1] for frame in buffer_frames) / len(buffer_frames)

            if average_rms > speech_volume and listening_enabled:
                if not recording:
                    print("Starting recording...")
                    recording = True
                    frames_since_noise = 0
                frames.extend(frame[0] for frame in buffer_frames)
                buffer_frames = []
            elif recording:
                if rms > speech_volume:
                    frames_since_noise = 0
                else:
                    frames_since_noise += 1

                frames.append(data)

                # Stop recording after silence persists
                if frames_since_noise > 30:
                    print("Stopping recording...")
                    recording = False
                    save_and_transcribe(frames, audio, client, audio_queue)
                    frames = []
    finally:
        stream.close()
        audio.terminate()


def save_and_transcribe(frames, audio, client, audio_queue):
    print("Transcribing...")
    file_path = "audio.wav"
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
    wf.writeframes(b''.join(frames))
    wf.close()

    with open(file_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    os.remove(file_path)
    print("Result org: ", result)
    print("Result text: ", result.text)
    audio_queue.put(result.text)

def main():
    api_key = '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'
    client = OpenAI(api_key=api_key)
    audio_queue = queue.Queue()
    speech_thread = threading.Thread(target=live_speech_to_text, args=(audio_queue, client))
    speech_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == '__main__':
    main()
