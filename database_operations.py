# Student Names:   Dylan Holmwood and Kristers Martukans
# Student Numbers: D21124331 and D21124318
# Date:            29th May 2024
# Module Title:    Final Year Project
# Module Code:     PROJ4004
# Supervisors:     Paula Kelly and Damon Berry
# Script Name:     database_operations.py
# Description:     This file handles the database operations including creating a connection, creating tables, inserting user profiles and facial embeddings, 
#                  retrieving and decrypting embeddings, and deleting old records. The script also ensures data security 
#                  by encrypting sensitive information before storing it in the SQLite database and decrypting it upon retrieval.
#                  Additionally, it manages the automatic deletion of old records and database cleanup on exit.


""" Libraries """
import sqlite3
import time
import os
import logging
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
import numpy as np

# Function to load encryption key from file
def load_encryption_key():
    with open('config.key', 'rb') as key_file:
        key = key_file.read()
    # Create a Fernet object with the loaded key
    return Fernet(key)

# Use Fernet object for encryption
cipher_suite = load_encryption_key()

# Function to create a connection to the SQLite database
def create_connection(db_file):
    """Create a connection to the SQLite database."""
    conn = None
    try:
        # Attempt to connect to the SQLite database file
        conn = sqlite3.connect(db_file, check_same_thread=False)
        conn.execute('PRAGMA foreign_keys = ON;')  # Enable foreign key constraints
        print("SQLite version:", sqlite3.version)  # Print SQLite version (for debugging)
    except Exception as e:
        print(e)  # Print any errors that occur during connection (for debugging)
    return conn

# Function to create tables in the database
def create_tables(conn):
    """Create tables as per the updated schema."""
    # Enable foreign key constraints
    conn.execute('PRAGMA foreign_keys = ON;')

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
        FOREIGN KEY (user_id) REFERENCES user_profiles (id) ON DELETE CASCADE
    );'''

    c = conn.cursor()  # Get cursor for executing SQL statements
    # Execute SQL statements to create tables
    c.execute(user_profiles_table_sql)  # Execute SQL to create user_profiles table
    c.execute(facial_embeddings_table_sql)  # Execute SQL to create facial_embeddings table
    conn.commit()  # Commit the transaction (save changes to the database)


# Function to insert a user profile into the database
def insert_user_profile(conn, name="Anonymous", age=None):
    """Insert a user profile into the database with name and age."""
    try:
        # Ensure age is converted to a string before encoding and encrypting
        age_str = str(age) if age is not None else "Unknown"
        
        # Encrypt sensitive data before insertion
        encrypted_name = cipher_suite.encrypt(name.encode())         # Encrypt name
        encrypted_age = cipher_suite.encrypt(age_str.encode())       # Encrypt age
        
        sql = 'INSERT INTO user_profiles (name, age) VALUES (?, ?)'  # SQL statement
        
        # Get cursor for executing SQL statements
        cur = conn.cursor()                 

        # Execute SQL statement to insert user profile
        cur.execute(sql, (encrypted_name, encrypted_age))
        conn.commit()                                                # Commit the transaction
        return cur.lastrowid                                         # Return the last inserted row ID
    except Exception as e:
        print(f"Error inserting data: {e}")

    

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

# Function to retrieve and decrypt all user embeddings from the database
def get_all_embeddings(conn):
    """Retrieve and decrypt all user embeddings from the database."""
    logging.info("Retrieving and decrypting all user embeddings from the database.")
    
    embeddings = []
    user_ids = []
    sql = "SELECT user_id, embedding FROM facial_embeddings"  # SQL query to retrieve embeddings
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

# Function to retrieve the name of a returning user by user ID
def get_returning_user_name(conn, user_id):
    """Retrieve the name of a returning user by user ID."""
    logging.info(f"Retrieving name for user ID: {user_id}")

    sql = "SELECT name FROM user_profiles WHERE id = ?"
    cur = conn.cursor()
    cur.execute(sql, (user_id,))
    result = cur.fetchone()

    if result:
        encrypted_name = result[0]
        try:
            # Decrypt the name
            decrypted_name = cipher_suite.decrypt(encrypted_name).decode('utf-8')  # decode from bytes to string
            logging.info("User name decrypted and retrieved successfully.")
            return decrypted_name
        except Exception as e:
            logging.error(f"Error decrypting name for user ID {user_id}: {e}")
            return None
    else:
        logging.warning("User name not found for the given user ID.")
        return None

# Function to delete user profiles older than a certain date
def delete_old_records(conn):
    """Delete user profiles older than a certain date and automatically delete associated facial embeddings."""
    try:
        while True:
            cur = conn.cursor()

            # Define the time threshold, e.g., delete records older than 3 minutes for testing
            time_threshold = datetime.now() - timedelta(minutes=2)

            # Convert the time threshold to a string in the format SQLite understands
            time_threshold_str = time_threshold.strftime('%Y-%m-%d %H:%M:%S')

            logging.info(f"Attempting to delete records older than {time_threshold_str}")

            # Delete user profiles older than the time threshold
            delete_profiles_sql = "DELETE FROM user_profiles WHERE created_at < ?"
            cur.execute(delete_profiles_sql, (time_threshold_str,))

            # Commit the transaction
            conn.commit()
            deleted_rows = cur.rowcount  # Number of rows deleted
            logging.info(f"{deleted_rows} old records deleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{deleted_rows} old records deleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Sleep for a set amount of time before checking again (e.g., 6 minutes)
            time.sleep(300)
    except Exception as e:
        logging.error(f"Error in record deletion thread: {e}")

# Function to delete the database file on exit
def delete_database_on_exit(db_path):
    """Delete the database file."""
    if os.path.exists(db_path):
        os.remove(db_path)
        print("Database deleted on exit.")
        logging.info("Database deleted on exit.")
    else:
        print("Database file does not exist.")


