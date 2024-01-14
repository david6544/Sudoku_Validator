#%%
import cv2
import numpy as np


def process_sudoku_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

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
cells = process_sudoku_image('sudoku.png')

# To visualize the extracted cells
for i, cell in enumerate(cells):
    cv2.imshow(f'Cell {i+1}', cell)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# %%
