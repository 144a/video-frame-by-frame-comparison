import cv2
import numpy as np
import time

def process_rectangles(norm_diff, norm_diff_color, num_rect_horizontal, num_rect_vertical, threshold):
    height, width = norm_diff.shape
    # Calculate the dimensions of each rectangle
    rect_width = width // num_rect_horizontal
    rect_height = height // num_rect_vertical

    # Adjust width and height to be exactly divisible by the number of rectangles
    adjusted_width = rect_width * num_rect_horizontal
    adjusted_height = rect_height * num_rect_vertical

    # Crop or pad the image to match the adjusted dimensions
    norm_diff = norm_diff[:adjusted_height, :adjusted_width]

    # Reshape norm_diff to a 4D array where each block can be addressed separately
    reshaped_diff = norm_diff.reshape(num_rect_vertical, rect_height, num_rect_horizontal, rect_width)
    
    # Compute the mean across the height and width of each block
    block_means = reshaped_diff.mean(axis=(1, 3))
    
    # Prepare to draw rectangles if the mean exceeds the threshold
    for i in range(num_rect_vertical):
        for j in range(num_rect_horizontal):
            if block_means[i, j] > threshold:
                # Calculate the top left corner and the bottom right corner of the rectangle
                top_left = (j * rect_width, i * rect_height)
                bottom_right = ((j + 1) * rect_width, (i + 1) * rect_height)
                # Draw a red rectangle on norm_diff_color
                cv2.rectangle(norm_diff_color, top_left, bottom_right, (0, 0, 255), 2)

    return block_means

def main(real_time=False, fast_process=False, display_output=True, rectangle_processing=False, scale_factor=0.5, num_rect_horizontal=32, num_rect_vertical=32, threshold=15):
    # Load the video
    cap = cv2.VideoCapture('test.mp4')
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the frame rate of the video if real_time is True
    if real_time:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = 1 / fps  # Frame interval in seconds
    else:
        frame_interval = 0

    # Initialize frame storage
    frames = []

    # Read the first two frames to start
    for _ in range(2):
        ret, frame = cap.read()
        if not ret:
            print("Failed to read enough frames.")
            return
        frames.append(frame)

    last_frame_time = time.time()
    next_frame_time = time.time() + frame_interval  # Initialize next frame time

    # Process the video frame by frame
    while True:
        ret, current_frame = cap.read()
        if not ret:
            break

        frames.append(current_frame)

        # Calculate the difference and the average magnitude
        diff = cv2.absdiff(frames[-3], frames[-1])
        avg_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        norm_diff = cv2.normalize(avg_diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        norm_diff_color = cv2.cvtColor(norm_diff, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for consistency

        if rectangle_processing:
            # Process rectangles and get magnitude averages
            magnitude_averages = process_rectangles(norm_diff, norm_diff_color, num_rect_horizontal, num_rect_vertical, threshold)

        if display_output:
            # Resize images for display after processing
            resized_frame1 = cv2.resize(frames[-3], (0, 0), fx=scale_factor, fy=scale_factor)
            resized_diff = cv2.resize(norm_diff_color, (0, 0), fx=scale_factor, fy=scale_factor)
            resized_frame2 = cv2.resize(frames[-1], (0, 0), fx=scale_factor, fy=scale_factor)

            # Concatenate frames horizontally: first frame, difference, current frame
            combined_frame = cv2.hconcat([resized_frame1, resized_diff, resized_frame2])

            # Calculate FPS
            current_time = time.time()
            fps_display = 1 / (current_time - last_frame_time)
            last_frame_time = current_time
            fps_text = f"FPS: {fps_display:.2f}"
            cv2.putText(combined_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display the combined image
            cv2.imshow('Frame Comparison', combined_frame)

        # Remove the oldest frame to keep the buffer size constant
        frames.pop(0)

        # Wait for the correct time to display the next frame
        while time.time() < next_frame_time:
            time.sleep(0.001)  # Sleep in short intervals to stay responsive
        next_frame_time += frame_interval

        # Check for user input
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(real_time=False, fast_process=True, display_output=True, rectangle_processing=True, scale_factor=0.5)
