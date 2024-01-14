#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
model_name = 'digit_recognizer.h5'
loaded_model = tf.keras.models.load_model(model_name)

import numpy as np


def remove_lines(gray_img):
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Detect lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    lines = horizontal_lines + vertical_lines

    # Invert the image to make lines white and background black
    lines = cv2.bitwise_not(lines)

    # Combine the lines image with the original image
    cleaned_img = cv2.bitwise_and(gray_img, gray_img, mask=lines)

    return cleaned_img

# Load image
img = cv2.imread('sudoku.png')

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Now use gray_img for further processing
cleaned_img = remove_lines(gray_img)

# Process the image
cleaned_img = remove_lines(gray_img)

def process_sudoku_image(img):
    # Since img is already a grayscale image, we can directly apply thresholding
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find the largest contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour and get the bounding box
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    x, y, w, h = cv2.boundingRect(approx)

    # Extract and warp the grid
    grid = img[y:y+h, x:x+w]
    warped = cv2.resize(grid, (252, 252))  # Resize to 9 times 28

    # Extract and resize each cell to 28x28
    cell_size = warped.shape[0] // 9
    cells = []
    for i in range(9):
        for j in range(9):
            cell = warped[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_resized = cv2.resize(cell, (28, 28))
            cells.append(cell_resized)

    return cells

# Example usage
cells = process_sudoku_image(gray_img)

for cell in cells:
    plt.imshow(cell)

# Show the result
cv2.imwrite('Cleaned_sudoku.png', cleaned_img)


# %%
def preprocess_cell(cell):
     # Convert the cell to grayscale if it's not already
    if len(cell.shape) == 3:
        cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    else:
        cell_gray = cell
    # Assuming the cell is a grayscale image of size 28x28
    # Normalize pixel values to be between 0 and 1
    cell_normalized = cell_gray / 255.0
    # Reshape the cell for the model (if needed)
    # This depends on how your model was trained
    cell_reshaped = np.reshape(cell_normalized, (1, 28, 28, 1))  # This is the expected shape for most CNN models
    return cell_reshaped

predictions = []

for cell in cells:
    # Preprocess the cell
    preprocessed_cell = preprocess_cell(cell)

    # Use the model to predict the digit
    pred = loaded_model.predict(preprocessed_cell)
    
    # Get the digit with the highest probability
    digit = np.argmax(pred)
    
    # Append the predicted digit to the predictions list
    predictions.append(digit)

# Number of rows and columns in the grid
n_rows = 9
n_cols = 9

# Create a figure with subplots
plt.figure(figsize=(10, 10))  # Adjust the size as needed

for i in range(n_rows * n_cols):
    ax = plt.subplot(n_rows, n_cols, i + 1)
    
    # Display the cell image
    plt.imshow(cells[i], cmap='gray')
    
    # Remove axis ticks
    plt.xticks([])
    plt.yticks([])
    
    # Add the prediction as the title of the subplot
    plt.title(str(predictions[i]), fontsize=8)

plt.tight_layout()
plt.show()
# %%

from validate import validate_sudoku
# Assuming predictions is a list of 81 predicted digits
sudoku_grid = [predictions[i * 9:(i + 1) * 9] for i in range(9)]

for i in range(9):
    for j in range(9):
        print(sudoku_grid[i][j], end=" ")
        
# Pass the sudoku grid to the validation function
validation_result = validate_sudoku(sudoku_grid)

# Do something with the validation result
print("Validation Result:", validation_result)

# %%
