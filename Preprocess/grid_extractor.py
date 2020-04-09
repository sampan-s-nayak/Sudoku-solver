import cv2
import numpy as np
import argparse

def extract_cells(img):
    """
        takes image of sudoku square (no background) as input and returns an array containing the 81 cells as output
            img: image of the sudoku puzzle after   
                 perspective transform
    """
    print("extracting each individual cell in the grid.....")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    (height,width) = (img.shape[0],img.shape[1])

    # divide the sudoku puzzle into 81 cells
    step_size_h = height//9
    step_size_w = width//9

    # store the cells in grid
    grid = []
    for i in range(0,height+1,step_size_h):
        for j in range(0,width+1,step_size_w):
            if(j+step_size_w<=width):
                grid.append(img[i:i+step_size_h,j:j+step_size_w])

    # return the 81 cells
    return grid[:81]