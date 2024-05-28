# Student Names:   Dylan Holmwood and Kristers Martukans
# Student Numbers: D21124331 and D21124318
# Date:            21st May 2024
# Module Title:    Final Year Project
# Module Code:     PROJ4004
# Supervisors:     Paula Kelly and Damon Berry 
# Script Name:     import_libraries.py
# Description:     This script imports all necessary libraries for various functionalities including 
#                  computer vision, audio processing, threading, and AI model interaction.

import cv2                                     # OpenCV for computer vision tasks such as image capture and processing
import time                                    # Time-related functions for delays and timestamping
import threading                               # Threading for concurrent execution
import queue                                   # Queue for thread-safe data exchange between threads
from facenet_pytorch import InceptionResnetV1  # FaceNet model for facial recognition
import torchvision.transforms as transforms    # PyTorch transforms for image preprocessing
from PIL import Image                          # Pillow for image manipulation and conversion
import mediapipe as mp                         # MediaPipe for face detection
import torch                                   # PyTorch for tensor operations and deep learning model handling
import os                                      # OS module for file and directory operations
from openai import OpenAI                      # OpenAI API client for interacting with OpenAI models
import logging                                 # Logging for tracking events and debugging
import cProfile                                # cProfile for profiling and performance analysis
import sys                                     # System-specific parameters and functions
import numpy as np                             # NumPy for numerical operations on arrays
import audioop                                 # Audioop for basic audio operations
import pyaudio                                 # PyAudio for audio stream handling
import wave                                    # Wave for reading and writing .wav audio files
from playsound import playsound                # Playsound for playing audio files
import re                                      # Regular expressions for string matching and manipulation
from scipy.signal import butter, lfilter       # Signal processing functions for filtering audio signals

