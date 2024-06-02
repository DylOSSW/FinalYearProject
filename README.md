# Integrated Facial Recognition and Voice-Enabled Chatbot

## Project Overview

This project integrates facial recognition with a voice-enabled chatbot to create an interactive user experience. The system identifies users through facial recognition, collects demographic information, and engages in conversations using OpenAI's GPT models. It provides personalized responses and uses text-to-speech (TTS) for audio interactions.

## Table of Contents

1. [Project Setup](#project-setup)
2. [System Architecture](#system-architecture)
3. [Usage Instructions](#usage-instructions)
4. [Key Features](#key-features)
5. [Modules Overview](#modules-overview)
6. [API Keys and Configuration](#api-keys-and-configuration)
7. [Known Issues](#known-issues)
8. [Future Enhancements](#future-enhancements)

## Video Demonstration

Youtube Link: https://youtu.be/X12ZuIElGIY

## Project Setup

1. **Clone the Repository**
   ```sh
   git clone https://github.com/yourusername/facial-voice-chatbot.git
   cd facial-voice-chatbot

## System Architecture

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
- MTCNN
- Facenet
- SQLite (for local data storage)
- TensorFlow or PyTorch (for deep learning models)
- SpeechRecognition and pyttsx3 (for speech recognition and synthesis)

## Installation

Ensure your Raspberry Pi is set up and connected to the internet. Then, follow these steps:

1. **Update and Upgrade Your Pi:**

   ```bash
   sudo apt-get update
   sudo apt-get upgrade


## Jira Issues
- Testing Jira Integration [FYP-16].

