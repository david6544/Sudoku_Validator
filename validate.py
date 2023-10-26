#Validating Sudoku from a 2d array
import numpy as np
from cnn import *


def validate(array: list) :

    #Check rows and columns
    for rows in range(9) :
        rowCount = [0,0,0,0,0,0,0,0,0]
        colCount = [0,0,0,0,0,0,0,0,0]
        for cols in range(9): 
            rowCount[array[rows][cols]-1] += 1
            colCount[array[cols][rows]-1] += 1
            print(array[rows][cols],end=" ")
            if (rowCount[array[rows][cols]-1] > 1 or colCount[array[cols][rows]-1] > 1) :
                print("\nINVALID SUDOKU IN A ROW OR COLUMN")
                return
        print("\n")
        
    
    #Check 3x3
    for i in range (3) :
        for j in range (3) :
            threeCount = [0,0,0,0,0,0,0,0,0]
            for k in range (3) :
                for l in range (3) :
                    threeCount[array[i*3+k][j*3+l] - 1] += 1
                if (threeCount[array[i*3+k][j*3+l] - 1] > 1) :
                    print("\n INVALID SUDOKU IN A 3x3")
                    return
    
    print("VALID SUDOKU")

#Drivers
arr = [[4,3,5,2,6,9,7,8,1],[6,8,2,5,7,1,4,9,3],[1,9,7,8,3,4,5,6,2],[8,2,6,1,9,5,3,4,7],[3,7,4,6,8,2,9,1,5],[9,5,1,7,4,3,6,2,8],[5,1,9,3,2,6,8,7,4],[2,4,8,9,5,7,1,3,6],[7,6,3,4,1,8,2,5,9]]
arr2 = [[4,3,5,2,6,9,7,8,1],[6,8,2,5,7,1,4,9,3],[1,9,7,8,3,4,5,2,2],[8,2,6,1,9,5,3,4,7],[3,7,4,6,8,2,9,1,5],[9,5,1,7,4,3,6,2,8],[5,1,9,3,2,6,8,7,4],[2,4,8,9,5,7,1,3,6],[7,6,3,4,1,8,2,5,9]]

validate(arr)
validate(arr2)