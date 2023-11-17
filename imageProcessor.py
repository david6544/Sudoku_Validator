import cv2
import numpy as np
import matplotlib.pyplot as plt

#load the image
img = cv2.imread('sudoku.png')


#greyscale the image
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_grey, (5,5), 0)
thresh = cv2.adaptiveThreshold(img_blur, 255, 1, 1, 11, 2)

#Detect Sudoku Grid
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
polygon = contours[0]

# Approximate the contour to a 4-point polygon
epsilon = 0.02 * cv2.arcLength(polygon, True)
approx_polygon = cv2.approxPolyDP(polygon, epsilon, True)

# Ensure the approximated polygon has 4 points
if len(approx_polygon) != 4:
    raise ValueError("The approximated polygon does not have 4 points. Image might not contain a clear Sudoku grid.")

transform_mat = cv2.getPerspectiveTransform(np.float32(approx_polygon), np.float32([[0,0],[0,250],[250,250],[250,0]]))
top_down = cv2.warpPerspective(img, transform_mat, (250,250), cv2.INTER_LINEAR)

#extract cells

cell_size = int(top_down.shape[0] / 9)
cells = []

for i in range(9):
    for j in range(9):
        cell = top_down[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
        cells.append(cell)
                 
for cell in cells:
    plt.imshow(cell, cmap='gray')
    plt.show()
