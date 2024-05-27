# Student Names:   Dylan Holmwood and Kristers Martukans
# Student Numbers: D21124331 and D21124318
# Date:            29th May 2024
# Module Title:    Final Year Project
# Module Code:     PROJ4004
# Supervisors:     Paula Kelly and Damon Berry
# Script Name:     audio_files.py
# Description:     This dictionary maps text phrases to their corresponding audio file names. These audio files are used in an application 
#                  to provide spoken feedback and instructions to the user during the session. The keys in the dictionary are the text 
#                  phrases, and the values are the names of the audio files that contain the spoken version of those phrases.

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
    "Okay, so I will now begin to create a profile for you.": "profile_initiation.mp3",
    "Oh, you don't have a profile? Let's get one setup for you!": "no_profile.mp3",
    ("This statement outlines the data privacy and security protocols employed during this session. "
     "Facial features are collected for the purpose of facial recognition, and demographic information "
     "such as name and age are used to personalize the user experience. All data collected is encrypted "
     "and securely stored. Importantly, this data is retained for a maximum duration of one hour before "
     "being automatically deleted, thereby ensuring the privacy and security of user information. "
     "Furthermore, it's important to understand that during this session, your voice data will be sent to OpenAI, "
     "a third-party provider, for speech-to-text (speech recognition) conversion. OpenAI processes this data solely "
     "for transcription purposes and returns the transcription result, adhering to a strict zero data retention policy "
     "with its Whisper transcription service. This ensures that your privacy is upheld and no voice inputs are stored."):"data_usage_and_privacy_statement.mp3",  
    "During this session, we'll capture and analyze your facial features and demographic data to personalize your experience. Would you like to listen to our full data handling statement?":
    "brief_data_statement.mp3",
    "I couldn't understand the name, please try again.":"unclear_name_error.mp3",
    "Sorry I couldn't make out your age, could you please say it again?.": "unclear_age_error.mp3",
    "Please listen to the following instructions.": "instruction_intro.mp3",
    "Thank you, for providing your information, your profile is now complete!": "profiling_completed_message.mp3",
    "I will now peform some calibration to capture your facial features from a few angles for better accuracy.": "calibration_message.mp3",
    "Sorry I could not make out your name, I am going to exit profiling.": "no_name.mp3",
    "Sorry I could not make out your age, I am going to exit profiling.": "no_age.mp3"
}