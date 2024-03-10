# AI-Integrated Face Recognition and Tracking System

## Project Overview

This project aims to develop an AI-integrated face recognition and tracking system using a Raspberry Pi. It is divided into two main parts: User Profiling and Face Detection, and Face Recognition.

### User Profiling and Face Detection

- Detect a user's face when it enters the frame.
- Request user consent to record facial features and demographic data.
- Upon consent, capture the user's name and facial landmarks/embeddings.
- Prompt the user for a question.
- Store the facial/demographic data and the question in a database.

### Face Recognition

- Extract facial features/demographic data from the database.
- Compare live embeddings with those stored to perform facial recognition.
- Send the voice transcript (question) to ChatGPT to receive an answer.
- Provide a personalized interaction by addressing the user by their name and delivering the answer to their question.

## Requirements

### Hardware

- Raspberry Pi (preferably 4 or newer) with a camera module.

### Software

- Python 3.x
- OpenCV
- DLib
- SQLite (for local data storage)
- TensorFlow or PyTorch (for deep learning models)
- SpeechRecognition and pyttsx3 (for speech recognition and synthesis)

## Installation

Ensure your Raspberry Pi is set up and connected to the internet. Then, follow these steps:

1. **Update and Upgrade Your Pi:**

   ```bash
   sudo apt-get update
   sudo apt-get upgrade
