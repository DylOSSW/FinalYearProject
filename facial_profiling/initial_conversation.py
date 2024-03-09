import speech_recognition as sr
import pyttsx3

def text_to_speech(text):
    """Convert text to speech."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def listen_for_speech():
    """Listen for a single speech input and return the text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            text_to_speech("Sorry, I did not understand that.")
            return ""
        except sr.RequestError as e:
            text_to_speech(f"Could not request results; {e}")
            return ""

def ask_for_consent():
    """Ask the user for consent and return their response."""
    text_to_speech("Do you consent to this session and to have your facial features and demographic data captured for the session? This data will be used to perform facial recognition and provide a personalized experience. Please say yes to consent or no to decline.")
    return listen_for_speech().lower()

def main():
    consent = ask_for_consent()
    if "yes" in consent:
        text_to_speech("Thank you for your consent. Please tell me your name.")
        name = listen_for_speech()
        text_to_speech(f"Hello, {name}. What question would you like to ask?")
        question = listen_for_speech()

        print(question)

    else:
        text_to_speech("You have declined consent. The session will not proceed.")

if __name__ == "__main__":
    main()
