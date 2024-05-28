# Name:            Dylan Holmwood
# Student Number:  D21124331
# Collaborator:    Kristers Martukans
# Date:            28th March 2024
# Module Title:    Final Year Project
# Module Code:     PROJ4004
# Supervisors:     Paula Kelly and Damon Berry 
# Script Name:     database_operations.py
# Description: 

import sqlite3
import speech_recognition as sr
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import tensorflow as tf
import mediapipe as mp
from cryptography.fernet import Fernet

# Function to load encryption key from file
def load_encryption_key():
    with open('config.key', 'rb') as key_file:
        key = key_file.read()
        print(key)  # Print the loaded key (for debugging)
    # Create a Fernet object with the loaded key
    return Fernet(key)

# Use the Fernet object for encryption
encryption_tool = load_encryption_key()

# Function to create a connection to the SQLite database
def create_connection(db_file):
    """Create a connection to the SQLite database."""
    conn = None
    try:
        # Attempt to connect to the SQLite database file
        conn = sqlite3.connect(db_file, check_same_thread=False)
        print("SQLite version:", sqlite3.version)  # Print SQLite version (for debugging)
    except Exception as e:
        print(e)  # Print any errors that occur during connection (for debugging)
    return conn

# Function to create database tables
def create_tables(conn):
    """Create tables as per the updated schema."""
    # SQL statements to create user_profiles and facial_embeddings tables
    user_profiles_table_sql = '''
    CREATE TABLE IF NOT EXISTS user_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER, 
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
    );'''

    facial_embeddings_table_sql = '''
    CREATE TABLE IF NOT EXISTS facial_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        embedding BLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
        FOREIGN KEY (user_id) REFERENCES user_profiles (id)
    );'''

    c = conn.cursor()  # Get cursor for executing SQL statements
    # Execute SQL statements to create tables
    c.execute(user_profiles_table_sql)
    c.execute(facial_embeddings_table_sql)
    conn.commit()  # Commit the transaction

# Function to insert user profile into the database
def insert_user_profile(conn, name="Anonymous", age=None):
    """Insert a user profile into the database with name and age."""
    try:
        # Encrypt sensitive data before insertion
        print(name)  # Print the name (for debugging)
        encrypted_name = encryption_tool.encrypt(name.encode())  # Encrypt name
        print(encrypted_name)  # Print encrypted name (for debugging)
        encrypted_age = encryption_tool.encrypt(name.encode())  # Encrypt age
        sql = 'INSERT INTO user_profiles (name, age) VALUES (?, ?)'  # SQL statement
        cur = conn.cursor()  # Get cursor for executing SQL statements
        # Execute SQL statement to insert user profile
        cur.execute(sql, (encrypted_name, encrypted_age))
        conn.commit()  # Commit the transaction
        return cur.lastrowid  # Return the last inserted row ID
    except Exception as e:
        print(f"Error inserting data: {e}")  # Print any errors (for debugging)

# Function to insert facial embedding into the database
def insert_embedding(conn, user_id, embedding):
    """Insert facial embedding into the database."""
    try:
        # Convert the embedding tensor to a NumPy array and then to bytes
        embedding_bytes = embedding.tobytes()

        # Encrypt the embedding bytes
        encrypted_embedding_bytes = encryption_tool.encrypt(embedding_bytes)

        sql = "INSERT INTO facial_embeddings (user_id, embedding) VALUES (?, ?)"  # SQL statement
        cur = conn.cursor()  # Get cursor for executing SQL statements
        # Execute SQL statement to insert facial embedding
        cur.execute(sql, (user_id, encrypted_embedding_bytes))
        conn.commit()  # Commit the transaction
    except Exception as e:
        print(f"Error inserting embedding: {e}")  # Print any errors (for debugging)
