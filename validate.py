#%%
#Validating Sudoku from a 2d array
import numpy as np


def validate_sudoku(array: list):
    
    errors = []  # List to store error locations
    # Check rows and columns
    for i in range(9):
        rowCount = [0] * 9
        colCount = [0] * 9
        for j in range(9):
            rowCount[array[i][j] - 1] += 1
            colCount[array[j][i] - 1] += 1

        if any(count > 1 for count in rowCount):
                errors.append(('row', i))
        if any(count > 1 for count in colCount):
            errors.append(('column', i))
                
    # Check 3x3 grids
    for i in range(3):
        for j in range(3):
            threeCount = [0] * 9
            for k in range(3):
                for l in range(3):
                    threeCount[array[i*3+k][j*3+l] - 1] += 1

            if any(count > 1 for count in threeCount):
                print(f"INVALID SUDOKU: Duplicate in 3x3 Grid at Row {i*3+1} to {i*3+3}, Column {j*3+1} to {j*3+3}")
                return False

    print("VALID SUDOKU")
    return True


""" 
#Drivers
arr = [[4,3,5,2,6,9,7,8,1],[6,8,2,5,7,1,4,9,3],[1,9,7,8,3,4,5,6,2],[8,2,6,1,9,5,3,4,7],[3,7,4,6,8,2,9,1,5],[9,5,1,7,4,3,6,2,8],[5,1,9,3,2,6,8,7,4],[2,4,8,9,5,7,1,3,6],[7,6,3,4,1,8,2,5,9]]
arr2 = [[4,3,5,2,6,9,7,8,1],[6,8,2,5,7,1,4,9,3],[1,9,7,8,3,4,5,2,2],[8,2,6,1,9,5,3,4,7],[3,7,4,6,8,2,9,1,5],[9,5,1,7,4,3,6,2,8],[5,1,9,3,2,6,8,7,4],[2,4,8,9,5,7,1,3,6],[7,6,3,4,1,8,2,5,9]]

validate_sudoku(arr)
validate_sudoku(arr2) """
# %%
