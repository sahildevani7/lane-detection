import cv2
import numpy as np
import matplotlib as plt

def coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = coordinates(image, left_fit_average)
    right_line = coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


def canny_edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)   # Kernel size is 5x5
    canny = cv2.Canny(blur, 50, 150)
    return canny


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            # Reshaping all the lines to a 1D array.
            x1, y1, x2, y2 = line.reshape(4)
            # Draw a Blue Line(BGR in OpenCV)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])          
    # Triangle polygon because cv2.fillPoly expects an array of polygons.
    # Create a black mask to apply the above Triangle on.
    mask = np.zeros_like(image)
    # A complete white triangle polygon on a black mask.
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


cap = cv2.VideoCapture("test_video.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny_edge(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180,100, np.array([]), minLineLength=10, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    # Imposing the line_image on the original image
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    cv2.imshow('Result', combo_image)
    cv2.waitKey(1)
