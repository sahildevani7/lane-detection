# Road Lane Detection

This Python script performs basic lane detection on a video file. It processes video frames to identify and highlight the left and right lane lines using computer vision techniques like Canny edge detection and the Hough Transform. The script includes features for robustness, such as line averaging (using median), temporal smoothing, and coordinate validation to prevent crashes.

https://github.com/user-attachments/assets/a70182c0-f59b-4c10-af7a-6bbc219ee33b

## How It Works

This script processes video frames through the following pipeline:

1.  **Frame Input:** Reads the video frame-by-frame.
2.  **Edge Detection:** Applies Canny edge detection to identify potential lane lines within the frame.
3.  **Region Masking:** Isolates a predefined region of interest (ROI) where lanes are most likely to appear, filtering out irrelevant areas.
4.  **Line Detection:** Uses the Probabilistic Hough Transform on the masked edge image to detect straight line segments.
5.  **Lane Processing & Smoothing:**
    *   Filters detected line segments based on reasonable slope angles.
    *   Separates segments into left and right lane candidates.
    *   Calculates a robust representation of each lane line using the *median* slope/intercept of candidate segments.
    *   Applies temporal smoothing across frames to stabilize the detected lines and reduce jitter.
6.  **Line Generation & Validation:**
    *   Calculates the screen coordinates for the final smoothed lane lines.
    *   Performs multiple validation checks (finite values, coordinate bounds, safe integer ranges) to ensure stability before drawing.
7.  **Output:** Draws the validated lane lines onto the original frame and displays the resulting video.

## Future Development

This script serves as the core processing logic for a planned full-stack web application. The goal is to create a user-friendly interface where:

1.  **Frontend:** Users can upload their own video files through a web browser.
2.  **Backend (Python/Flask):** A Flask-based backend server will receive the uploaded video.
3.  **Processing:** The backend will utilize this lane detection script to process the video.
4.  **Output:** The resulting video, with the detected lane lines overlaid, will be sent back and displayed to the user on the frontend.
