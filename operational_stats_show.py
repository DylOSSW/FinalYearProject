import cv2
import time
import numpy as np
import threading
import psutil

def display_operational_stats(cap, frame, start_time):
    # Get operational statistics
    net_io = psutil.net_io_counters()
    bytes_sent = net_io.bytes_sent / (1024 ** 2)  # Convert to MB
    bytes_recv = net_io.bytes_recv / (1024 ** 2)  # Convert to MB
    thread_count = threading.active_count()
    cpu_usage = psutil.cpu_percent()
    fps = cap.get(cv2.CAP_PROP_FPS)  # Example, actual FPS calculation may vary
    ram_used = psutil.Process().memory_info().rss / (1024 ** 3)  # GB
    
    # Calculate uptime
    uptime_seconds = int(time.time() - start_time)
    uptime_hours = uptime_seconds // 3600
    uptime_minutes = (uptime_seconds % 3600) // 60
    uptime_seconds = uptime_seconds % 60

    # Format all statistics into one string
    stats_text = (
        f"Threads: {thread_count}\n"
        f"CPU: {cpu_usage}%\n"
        f"FPS: {fps}\n"
        f"RAM Used: {ram_used:.2f} GB\n"
        f"Net Sent: {bytes_sent:.2f} MB\n"
        f"Net Recv: {bytes_recv:.2f} MB\n"
        f"Uptime: {uptime_hours}h {uptime_minutes}m {uptime_seconds}s"
    )

    # Initialize vertical position for text
    vertical_pos = frame.shape[0] - 50  # Starting from bottom, going up
    line_height = 20  # Height of each line of text

    # Display each line of stats on the frame
    for line in stats_text.split('\n'):
        cv2.putText(frame, line, 
                    (10, vertical_pos), cv2.FONT_HERSHEY_COMPLEX, 
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
        vertical_pos -= line_height

    # Show the frames
    cv2.imshow('Developer View', frame)