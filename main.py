import cv2
import numpy as np

def coordinates(image, line_parameters):
    slope, intercept = line_parameters
    if slope == 0:
        print("Warning: Slope is zero in coordinates calculation.")
        return None 
        
    y1 = image.shape[0]  
    y2 = int(y1 * (3/5)) 
    
    x1 = (y1 - intercept) / slope
    x2 = (y2 - intercept) / slope
    
    if not all(np.isfinite([x1, x2])):
        print(f"Warning: Non-finite x calculated: x1={x1}, x2={x2}")
        return None

    return np.array([x1, y1, x2, y2]) 


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
        
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)    
        if x1 == x2:
            continue    
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    
    lines_to_return = []
    
    if left_fit:
        left_fit = [params for params in left_fit if all(np.isfinite(params))] 
        if left_fit:
            left_fit_average = np.average(left_fit, axis=0)
            if np.all(np.isfinite(left_fit_average)):
                left_line = coordinates(image, left_fit_average)
                if left_line is not None:
                    lines_to_return.append(left_line)
            
    if right_fit:
        right_fit = [params for params in right_fit if all(np.isfinite(params))]
        if right_fit:
            right_fit_average = np.average(right_fit, axis=0)
            if np.all(np.isfinite(right_fit_average)):
                right_line = coordinates(image, right_fit_average)
                if right_line is not None:
                     lines_to_return.append(right_line)

    return np.array(lines_to_return) if lines_to_return else None


def canny_edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        if lines.ndim == 1:
            lines = lines.reshape(1, 4) 

        for line in lines:
            if line is None: 
                continue
                
            if not np.all(np.isfinite(line)):
                print(f"Warning: Skipping line with non-finite values: {line}")
                continue

            try:
                x1, y1, x2, y2 = map(int, line) 
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10) 
            except ValueError:
                print(f"Warning: Skipping line due to ValueError on int conversion: {line}")
            except OverflowError:
                 print(f"Warning: Skipping line due to OverflowError on int conversion (coordinate too large?): {line}")
            except TypeError:
                 print(f"Warning: Skipping line due to TypeError on int conversion: {line}")

    return line_image


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)] 
    ], dtype=np.int32)        
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    if len(image.shape) == len(mask.shape) and image.shape[:2] == mask.shape[:2]:
        masked_image = cv2.bitwise_and(image, mask)
    else:
        masked_image = cv2.bitwise_and(image, image, mask=mask) 
    return masked_image


# --- Main Loop ---
cap = cv2.VideoCapture("test_video2.mp4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Reached end of video or error reading frame.")
        break
    
    try: 
        frame_copy = np.copy(frame) 

        canny_image = canny_edge(frame_copy)
        cropped_image = region_of_interest(canny_image)
        
        # Run Hough Lines detection
        lines = cv2.HoughLinesP(cropped_image, 
                                rho=2,              # Distance resolution in pixels
                                theta=np.pi/180,    # Angle resolution in radians
                                threshold=100,      # Minimum number of votes (intersections in Hough grid cell)
                                lines=np.array([]), # Placeholder, not needed
                                minLineLength=40,   # Minimum line length. 
                                maxLineGap=5        # Maximum allowed gap between points on the same line.
                               )
        
        averaged_lines = average_slope_intercept(frame_copy, lines) 
        
        line_image = display_lines(frame_copy, averaged_lines) 

        if line_image.shape == frame_copy.shape:
            combo_image = cv2.addWeighted(frame_copy, 0.8, line_image, 1, 1)
            cv2.imshow('Result', combo_image)
        else:
            print(f"Shape mismatch: frame {frame_copy.shape}, lines {line_image.shape}")
            cv2.imshow('Result', frame_copy) 

    except Exception as e:
        print(f"An error occurred during frame processing: {e}")
        import traceback
        traceback.print_exc()

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Video processing finished.")