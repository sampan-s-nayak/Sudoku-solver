import cv2
import numpy as np

def get_digit(img):
    """
        function to find the digit within each cell
            img: the cell in the sudoku grid
    """
    thresh = cv2.threshold(img, 0, 50,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cnts,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    digit = []
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # if the contour is sufficiently large, it must be a digit (assumtion)
        if w >= 10 and (h >= 15 and h <= 80) and ((x > 0 and x < img.shape[1])and (y > 0 and y < img.shape[0])):
           rect = [x,y,w,h]
           digit = extract_digit(img,rect)
    if len(digit):
        return digit
    else:
        return shrink_img(img)

def extract_digit(img,rect,offset=6):
    """
        extracts the part of the image enclosed within the rectangle specified by rect
            img: img of a cell in the sudoku grid
            rect: cordinates of the rectangle to be extracted
            offset: additional padding given to the rectangle
    """
    (x,y,w,h) = rect
    digit = img[y-offset:y+h+offset,x-offset:x+w+offset]
    return digit

def shrink_img(cell):
    """
        if no digit is found within the cell then the cell is mostly blank, so shrink the cell so that 
        remenants of the edges do not hinder our neural network from making the right prediction
    """
    height,width = cell.shape
    p1 = int(0.11 * height)
    p2 = int(0.11 * width )

    # shrinking the cell in all sides
    cell_new = cell[p1:height-p1,p2:width-p2]
    return cell_new