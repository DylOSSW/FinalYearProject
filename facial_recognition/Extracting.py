import cv2
import numpy as np
import sqlite3
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision.transforms as transforms
import threading
import queue

# Global queue for frames
frame_queue = queue.Queue()
# Event to signal the processing thread to terminate
stop_event = threading.Event()

# Database connection and model initialization
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Exception as e:
        print(e)
    return conn

def load_embeddings_from_db(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM facial_embeddings")
    embeddings = cursor.fetchall()
    loaded_embeddings = []
    for embedding in embeddings:
        try:
            # Directly load embedding as it is no longer encrypted
            emb_array = np.frombuffer(embedding[2], dtype=np.float32)  # Assuming embedding[2] is the embedding
            emb_tensor = torch.tensor(emb_array)
            loaded_embeddings.append((embedding[0], emb_tensor))
        except Exception as e:
            print(f"Error loading embedding for ID {embedding[0]}: {e}")
    return loaded_embeddings

def load_landmarks_from_db(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM facial_landmarks")
    landmarks = cursor.fetchall()
    loaded_landmarks = []
    for landmark in landmarks:
        try:
            # Assume landmarks are stored directly without encryption
            loaded_landmarks.append((landmark[1], landmark[2:7]))  # Adjust as necessary
        except Exception as e:
            print(f"Error loading landmarks for ID {landmark[1]}: {e}")
    return loaded_landmarks


def get_live_embedding(frame, mtcnn, facenet_model):
    try:
        img = Image.fromarray(frame)
        img_cropped = mtcnn(img)
        if img_cropped is not None:
            if img_cropped.ndim == 3:
                img_cropped = img_cropped.unsqueeze(0)
            embedding = facenet_model(img_cropped)
            return embedding
    except Exception as e:
        print(f"Error getting live embedding: {e}")
    return None

def compare_embeddings(embedding, loaded_embeddings, threshold=0.8):
    if embedding is not None:
        for id, loaded_embedding in loaded_embeddings:
            dist = (embedding - loaded_embedding).norm().item()
            if dist < threshold:
                return id
    return None

def process_frames(mtcnn, facenet_model, loaded_embeddings):
    while not stop_event.is_set() or not frame_queue.empty():
        if not frame_queue.empty():
            frame = frame_queue.get()
            embedding = get_live_embedding(frame, mtcnn, facenet_model)
            match_id = compare_embeddings(embedding, loaded_embeddings)
            if match_id is not None:
                print(f"Match found: {match_id}")
            else:
                print("No match found.")
            frame_queue.task_done()

def main():
    db_file = 'mtcnn.db'
    conn = create_connection(db_file)

    mtcnn = MTCNN(keep_all=True)
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

    loaded_embeddings = load_embeddings_from_db(conn)
    load_landmarks_from_db(conn)

    # Start the processing thread
    processing_thread = threading.Thread(target=process_frames, args=(mtcnn, facenet_model, loaded_embeddings))
    processing_thread.start()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Put the current frame into the queue for processing
        frame_queue.put(frame)

        cv2.imshow('Live Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Signal the processing thread to terminate and wait for it to finish
    stop_event.set()
    processing_thread.join()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()