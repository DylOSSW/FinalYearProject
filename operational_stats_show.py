# Student Names:   Dylan Holmwood and Kristers Martukans
# Student Numbers: D21124331 and D21124318
# Date:            21st May 2024
# Module Title:    Final Year Project
# Module Code:     PROJ4004
# Supervisors:     Paula Kelly and Damon Berry 
# Script Name:     operational_stats_show.py
# Description:     This file defines a function to display operational statistics on a video frame. The statistics include 
#                  thread count, CPU usage, frames per second (FPS), RAM usage, network data sent and received, and system uptime. 
#                  These statistics are overlaid on the video frame in real-time, providing a developer view of the system's 
#                  operational metrics.

import cv2
import time
import numpy as np
import threading
import psutil

def display_operational_stats(cap, frame, start_time):
    """ Display operational statistics on the video frame. """
    # Get operational statistics
    net_io = psutil.net_io_counters()
    bytes_sent = net_io.bytes_sent / (1024 ** 2)  # Convert to MB
    bytes_recv = net_io.bytes_recv / (1024 ** 2)  # Convert to MB
    thread_count = threading.active_count()
    cpu_usage = psutil.cpu_percent()
    fps = cap.get(cv2.CAP_PROP_FPS)  # Example, actual FPS calculation may vary
    ram_used = psutil.Process().memory_info().rss / (1024 ** 3)  # Convert to GB
    
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

    # Show the frame with the statistics
    cv2.imshow('Developer View', frame)
