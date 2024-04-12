import cv2
import sqlite3
import time
import threading
import queue
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import mediapipe as mp

# Add global variables to track the last matched user and the time of the last update.
last_matched_user_id = None
last_matched_time = time.time()
info_display_timeout = 5  # seconds

display_info = {"name": "Unknown", "age": "Unknown"}
display_timeout = 5  # seconds to wait before clearing display if no new match
last_display_update = time.time()  # track last update to display info


# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Global queue for frames
frame_queue = queue.Queue()
# Event to signal the processing thread to terminate
stop_event = threading.Event()
# This is a new global queue for user info messages.
user_info_queue = queue.Queue()

def create_connection(db_file):
    """Create a connection to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(db_file, check_same_thread=False)
        print("SQLite version:", sqlite3.version)
    except Exception as e:
        print(e)
    return conn

def get_all_embeddings(conn):
    """Retrieve all user embeddings from the database."""
    embeddings = []
    user_ids = []
    sql = "SELECT user_id, embedding FROM facial_embeddings"
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    for row in rows:
        user_ids.append(row[0])
        embedding = np.frombuffer(row[1], dtype=np.float32)
        embeddings.append(embedding)
    return user_ids, embeddings
   
def capture_embeddings(face_detection, facenet_model, image):
    """
    Detect faces using MediaPipe and capture facial embeddings using FaceNet.
    """
    # Convert the image color from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe Face Detection
    results = face_detection.process(image_rgb)
    
    embeddings = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                   int(bboxC.width * w), int(bboxC.height * h)
            cropped_face = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            face_image = Image.fromarray(cropped_face)
            face_image = face_image.resize((160, 160))
            face_tensor = transforms.ToTensor()(face_image).unsqueeze(0)
            
            # Generate the embedding using FaceNet model
            embedding = facenet_model(face_tensor)
            embeddings.append(embedding.cpu().detach().numpy().flatten())
    
    return embeddings
    
    
def find_closest_embedding(captured_embedding, embeddings, threshold=0.6):
    """Find the closest embedding in the database to the captured one, with a threshold for matching."""
    min_distance = float('inf')
    closest_embedding_index = -1
    for i, db_embedding in enumerate(embeddings):
        distance = np.linalg.norm(captured_embedding - db_embedding)
        if distance < min_distance:
            min_distance = distance
            closest_embedding_index = i
    if min_distance > threshold:  # If no embedding is close enough, return no match
        return -1
    return closest_embedding_index


def process_frames(detector, facenet_model, conn):
    global last_matched_user_id, last_matched_time
    user_ids, embeddings = get_all_embeddings(conn)
    
    
    while not stop_event.is_set() or not frame_queue.empty():
        if not frame_queue.empty():
            image = frame_queue.get()
            captured_embeddings = capture_embeddings(detector, facenet_model, image)
            if captured_embeddings:
                for captured_embedding in captured_embeddings:
                    captured_embedding_np = np.array(captured_embedding).flatten()
                    closest_index = find_closest_embedding(captured_embedding_np, embeddings)
                    if closest_index != -1:
                        current_time = time.time()
                        if last_matched_user_id != user_ids[closest_index] or (current_time - last_matched_time > info_display_timeout):
                            user_id = user_ids[closest_index]
                            last_matched_user_id = user_ids[closest_index]
                            last_matched_time = current_time
                            
                            # Fetch user details from the database
                            cur = conn.cursor()
                            cur.execute("SELECT name, age FROM user_profiles WHERE id = ?", (user_id,))
                            result = cur.fetchone()
                            if result:
                                name, age = result
                                print(f"Match found: Name - {name}, Age - {age}")  # Print match details
                                user_info_queue.put(result)  # Update the queue with new user info
                    else:
                        current_time = time.time()
                        # No match found, clear the display after a timeout
                        if last_matched_user_id is not None and (current_time - last_matched_time > info_display_timeout):
                            last_matched_user_id = None
                            last_matched_time = current_time
                            print("No match found.")  # Print no match
                            user_info_queue.put(("Unknown", "Unknown"))
            frame_queue.task_done()



def display_text_on_frame(frame, text, position=(50, 50), color=(0, 255, 0), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
    try:
        # Convert text to string
        text = str(text)
        # Convert color to tuple if not already
        if not isinstance(color, tuple):
            color = tuple(color)
        # Convert font to int if not already
        if not isinstance(font, int):
            font = getattr(cv2, font)
        cv2.putText(frame, text, position, font, font_scale, color, thickness)
    except Exception as e:
        print(f"Error displaying text on frame: {e}")

def main():
    global last_matched_user_id, last_matched_time, display_info, last_display_update
    db_file = 'face_recognition.db'
    conn = create_connection(db_file)
    
    if conn is not None:
        cap = cv2.VideoCapture(0)
        last_snapshot_time = time.time()
        
        with mp_face_detection.FaceDetection(min_detection_confidence=0.9) as face_detection:
            facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
            
            processing_thread = threading.Thread(target=process_frames, args=(face_detection, facenet_model, conn))
            processing_thread.start()

            try:
                while True:
                    ret, frame = cap.read()
                    if ret:
                        # Convert frame to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Detect faces
                        results = face_detection.process(frame_rgb)

                        # Draw bounding box around each detected face
                        if results.detections:
                            for detection in results.detections:
                                bboxC = detection.location_data.relative_bounding_box
                                ih, iw, _ = frame.shape
                                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                    int(bboxC.width * iw), int(bboxC.height * ih)
                                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)

                            current_time = time.time()
                            # Check for new user info and update display, or clear after timeout
                            if not user_info_queue.empty():
                                user_name, user_age = user_info_queue.get()
                                display_info["name"] = user_name
                                display_info["age"] = user_age
                                last_display_update = current_time
                            elif display_info["name"] and (current_time - last_display_update > display_timeout):
                                # Clear display info after timeout
                                display_info = {"name": "Unknown", "age": "Unknown"}

                        # Update display if there is info
                        if display_info["name"]:
                            display_text = f"Name: {display_info['name']}, Age: {display_info['age']}"
                            display_text_on_frame(frame, display_text)

                        cv2.imshow('Face Detection', frame)

                        current_time = time.time()
                        # Check if 5 seconds have passed since the last snapshot
                        if current_time - last_snapshot_time >= 5:
                            last_snapshot_time = current_time
                            # Put only the current frame into the queue for processing
                            frame_queue.put(frame)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

            finally:
                stop_event.set()
                processing_thread.join()
                conn.close()
                cap.release()
                cv2.destroyAllWindows()
    else:
        print("Failed to create a database connection.")

if __name__ == '__main__':
    main()