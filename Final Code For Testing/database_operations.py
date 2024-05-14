# Student Names:   Dylan Holmwood and Kristers Martukans
# Student Numbers: D21124331 and D21124318
# Date:            21st May 2024
# Module Title:    Final Year Project
# Module Code:     PROJ4004
# Supervisors:     Paula Kelly and Damon Berry 
# Script Name:     MainSystem.py
# Description:     This file serves as the.....

import sqlite3

from cryptography.fernet import Fernet
from datetime import datetime, timedelta
import time
import os
import logging
import numpy as np

# Function to load encryption key from file
def load_encryption_key():
    with open('config.key', 'rb') as key_file:
        key = key_file.read()
        print(key)  # Print the loaded key (for debugging)
    # Create a Fernet object with the loaded key
    return Fernet(key)

# Use the Fernet object for encryption
cipher_suite = load_encryption_key()

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
    c.execute(user_profiles_table_sql)  # Execute SQL to create user_profiles table
    c.execute(facial_embeddings_table_sql)  # Execute SQL to create facial_embeddings table
    conn.commit()  # Commit the transaction (save changes to the database)


'''# Function to insert user profile into the database
def insert_user_profile(conn, name="Anonymous", age=None):
    """Insert a user profile into the database with name and age."""
    try:

        sql = 'INSERT INTO user_profiles (name, age) VALUES (?, ?)'  # SQL statement
        cur = conn.cursor()  # Get cursor for executing SQL statements
        # Execute SQL statement to insert user profile
        cur.execute(sql, (name, age))
        conn.commit()  # Commit the transaction
        return cur.lastrowid  # Return the last inserted row ID
    except Exception as e:
        print(f"Error inserting data: {e}")  # Print any errors (for debugging)

# Function to insert facial embedding into the database
def insert_embeddings(conn, user_id, embedding):
    """Insert facial embedding into the database."""
    try:
        # Convert the embedding tensor to a NumPy array and then to bytes
        embedding_bytes = embedding.tobytes()
        sql = "INSERT INTO facial_embeddings (user_id, embedding) VALUES (?, ?)"  # SQL statement
        cur = conn.cursor()  # Get cursor for executing SQL statements
        # Execute SQL statement to insert facial embedding
        cur.execute(sql, (user_id, embedding_bytes))
        conn.commit()  # Commit the transaction
    except Exception as e:
        print(f"Error inserting embedding: {e}")  # Print any errors (for debugging)'''

# Function to insert user profile into the database
def insert_user_profile(conn, name="Anonymous", age=None):
    """Insert a user profile into the database with name and age."""
    try:
        # Encrypt sensitive data before insertion
        print(name)  # Print the name (for debugging)
        encrypted_name = cipher_suite.encrypt(name.encode())  # Encrypt name
        print(encrypted_name)  # Print encrypted name (for debugging)
        encrypted_age = cipher_suite.encrypt(age.encode())  # Encrypt age
        sql = 'INSERT INTO user_profiles (name, age) VALUES (?, ?)'  # SQL statement
        cur = conn.cursor()  # Get cursor for executing SQL statements
        # Execute SQL statement to insert user profile
        cur.execute(sql, (encrypted_name, encrypted_age))
        conn.commit()  # Commit the transaction
        return cur.lastrowid  # Return the last inserted row ID
    except Exception as e:
        print(f"Error inserting data: {e}")  # Print any errors (for debugging)

# Function to insert facial embedding into the database
def insert_embeddings(conn, user_id, embedding):
    """Insert facial embedding into the database."""
    try:
        # Convert the embedding tensor to a NumPy array and then to bytes
        embedding_bytes = embedding.tobytes()

        # Encrypt the embedding bytes
        encrypted_embedding_bytes = cipher_suite.encrypt(embedding_bytes)

        sql = "INSERT INTO facial_embeddings (user_id, embedding) VALUES (?, ?)"  # SQL statement
        cur = conn.cursor()  # Get cursor for executing SQL statements
        # Execute SQL statement to insert facial embedding
        cur.execute(sql, (user_id, encrypted_embedding_bytes))
        conn.commit()  # Commit the transaction
    except Exception as e:
        print(f"Error inserting embedding: {e}")  # Print any errors (for debugging)

'''def get_all_embeddings(conn):
    """Retrieve all user embeddings from the database."""
    logging.info("Retrieving all user embeddings from the database.")

    embeddings = []
    user_ids = []
    sql = "SELECT user_id, embedding FROM facial_embeddings"
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    
    for row in rows:
        user_id = row[0]
        user_ids.append(user_id)
        embedding_bytes = row[1]
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        embeddings.append(embedding)

    logging.info(f"Retrieved {len(embeddings)} embeddings for {len(user_ids)} users.")
    return user_ids, embeddings'''


def get_all_embeddings(conn):
    """Retrieve and decrypt all user embeddings from the database."""
    logging.info("Retrieving and decrypting all user embeddings from the database.")
    
    embeddings = []
    user_ids = []
    sql = "SELECT user_id, embedding FROM facial_embeddings"  # Adjust if more columns are needed
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    
    for row in rows:
        user_id = row[0]
        user_ids.append(user_id)
        
        # Decrypting the embedding
        try:
            decrypted_embedding_bytes = cipher_suite.decrypt(row[1])
            embedding = np.frombuffer(decrypted_embedding_bytes, dtype=np.float32)
            embeddings.append(embedding)
        except Exception as e:
            logging.error(f"Error decrypting embedding for user ID {user_id}: {e}")
            # Optionally log or print part of the problematic data for debugging
            logging.debug(f"Data sample (first 20 bytes): {row[1][:20]}")
    
    logging.info(f"Retrieved and decrypted {len(embeddings)} embeddings for {len(user_ids)} users.")
    return user_ids, embeddings

'''def get_returning_user_name(conn, user_id):
    logging.info(f"Retrieving name for user ID: {user_id}")

    sql = "SELECT name FROM user_profiles WHERE id = ?"
    cur = conn.cursor()
    cur.execute(sql, (user_id,))
    result = cur.fetchone()

    if result:
        logging.info("User name retrieved successfully.")
        return result[0]
    else:
        logging.warning("User name not found for the given user ID.")
        return None'''

def get_returning_user_name(conn, user_id):
    logging.info(f"Retrieving name for user ID: {user_id}")

    sql = "SELECT name FROM user_profiles WHERE id = ?"
    cur = conn.cursor()
    cur.execute(sql, (user_id,))
    result = cur.fetchone()

    if result:
        encrypted_name = result[0]
        try:
            # Assuming the encrypted_name is stored in a format that needs decryption
            decrypted_name = cipher_suite.decrypt(encrypted_name).decode('utf-8')  # decode from bytes to string
            logging.info("User name decrypted and retrieved successfully.")
            return decrypted_name
        except Exception as e:
            logging.error(f"Error decrypting name for user ID {user_id}: {e}")
            return None
    else:
        logging.warning("User name not found for the given user ID.")
        return None

def delete_old_records(conn):
    """Delete user profiles and their associated facial embeddings older than a certain date."""
    try:
        while True:
            cur = conn.cursor()
            # Define the time threshold, here we delete records older than 30 days for demonstration
            time_threshold = datetime.now() - timedelta(minutes=3)
            
            # Convert the time threshold to a string in the format SQLite understands
            time_threshold_str = time_threshold.strftime('%Y-%m-%d %H:%M:%S')

            # First, delete facial embeddings associated with user profiles that are going to be deleted
            delete_embeddings_sql = """
                DELETE FROM facial_embeddings 
                WHERE user_id IN (
                    SELECT id FROM user_profiles WHERE created_at < ?
                )
            """
            cur.execute(delete_embeddings_sql, (time_threshold_str,))

            # Then, delete the user profiles older than the time threshold
            delete_profiles_sql = "DELETE FROM user_profiles WHERE created_at < ?"
            cur.execute(delete_profiles_sql, (time_threshold_str,))
            
            conn.commit()
            print(f"Old records deleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logging.info(f"Old records deleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            
            # Sleep for a set amount of time before checking again, e.g., 1 day (86400 seconds)
            time.sleep(40)
    except Exception as e:
        print(f"Error in record deletion thread: {e}")


def delete_database_on_exit(db_path):
    """Delete the database file."""
    if os.path.exists(db_path):
        os.remove(db_path)
        print("Database deleted on exit.")
        logging.info("Database deleted on exit.")
    else:
        print("Database file does not exist.")


