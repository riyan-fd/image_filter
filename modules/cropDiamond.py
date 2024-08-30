import cv2
import numpy as np
# import requests
# from io import BytesIO
# import json
# import os
# import csv
# from urllib.parse import urlparse, parse_qs
# import config
# from modules.display_output import display_image
# from matplotlib import pyplot as plt
# from modules.image_helpers import make_square

def unsharp_mask(image, sigma=1.0, strength=2.0):
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)

    # Apply unsharp mask only to l (luminance) channel
    blurred_l = cv2.GaussianBlur(l, (0, 0), sigma)
    sharpened_l = cv2.addWeighted(l, 1.0 + strength, blurred_l, -strength, 0)

    # Merge channels and convert back to BGR
    sharpened_lab = cv2.merge([sharpened_l, a, b])
    sharpened_bgr = cv2.cvtColor(sharpened_lab, cv2.COLOR_Lab2BGR)
    return sharpened_bgr


def find_average_color(image, corner='tl', square_size=5):
    if corner == 'tl':  # top-left
        region = image[:square_size, :square_size]
    elif corner == 'tr':  # top-right
        region = image[:square_size, -square_size:]
    elif corner == 'bl':  # bottom-left
        region = image[-square_size:, :square_size]
    elif corner == 'br':  # bottom-right
        region = image[-square_size:, -square_size:]
    
    avg_color = np.mean(region, axis=(0, 1)).astype(int)
    return avg_color


def extract_shape(image):
# Load the image using OpenCV
    # image = cv2.imread(image_url)
    
# Check if the image was loaded successfully
    if image is None:
        print(f"Error: Could not load image from {image_url}")
    # Load and display the original image
    
    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahed_l = clahe.apply(l_channel)
    
    # Merge the CLAHE enhanced L channel with the original A and B channel
    lab_clahed_image = cv2.merge((clahed_l, a_channel, b_channel))
    
    # Convert back to BGR color space
    bgr_clahed_image = cv2.cvtColor(lab_clahed_image, cv2.COLOR_LAB2BGR)
    
    # Convert the enhanced image to Grayscale
    gray = cv2.cvtColor(bgr_clahed_image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges
    edges = cv2.Canny(blurred, 50, 150,L2gradient = True)

    # Morphological Closing Operation
    kernel = np.ones((3,3), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel, iterations=3)
    
    # Find contours
    contours, _ = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area (tweak the area value as per your requirement)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10000]
    
    if len(filtered_contours) < 1:
        return None
    # filtered_contours=contours
    if filtered_contours:
        # Assuming the diamond contour is the largest among all extracted contours
        largest_contour = max(contours, key=cv2.contourArea)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add margin to the bounding box
        margin = 10  # Adjust as needed
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)

        # Crop the image using the bounding box with margin
        cropped_image = image[y:y+h, x:x+w]
    # images=[image,cropped_image]
    # display_images(images)     
    return cropped_image

def unsharp_mask(image, sigma=1.0, strength=0.20):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened

def make_square(image, desired_size=500):
    h, w, _ = image.shape
    max_dim = max(h, w)
    canvas = np.zeros((max_dim, max_dim, 3), dtype="uint8")

    x_offset = (max_dim - w) // 2
    y_offset = (max_dim - h) // 2
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = image

    # Get average colors for corners
    tl = find_average_color(image, 'tl')
    tr = find_average_color(image, 'tr')
    bl = find_average_color(image, 'bl')
    br = find_average_color(image, 'br')

    # Apply corner colors to canvas border
    canvas[:y_offset, :] = tl  # Apply to top border
    canvas[y_offset+h:, :] = bl  # Apply to bottom border
    canvas[:, :x_offset] = tl  # Apply to left border
    canvas[:, x_offset+w:] = tr  # Apply to right border
    
    return cv2.resize(canvas, (desired_size, desired_size))

def crop_image(image):
     
    cropped=extract_shape(image)

    if( cropped is None ):
        print( f'Failed to process image' )
        return 'failed to process image'

    sharpened = unsharp_mask(cropped)
    # cv2.imwrite(output_path, sharpened)

    # resized_cropped = cv2.resize(sharpened, (500, 500))
    resized_cropped = make_square(sharpened, 500)
    return resized_cropped
    # try:
    #     # Attempt to save the image
    #     success = cv2.imwrite(output_path, resized_cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
    #     if success:
    #         print(f"Test image successfully saved to {output_path}")
    #     else:
    #         print("Error: Failed to save the test image.")
    # except cv2.error as e:
    #     # Handle OpenCV specific errors
    #     print(f"OpenCV error: {e}")
    # except Exception as e:
    #     # Handle other exceptions
    #     print(f"Unexpected error: {e}")