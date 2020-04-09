import cv2
import numpy as np

""" 
this script deals with finding the sudoku square (and removing the background)
"""

def extract_grid(img_path):
    """
        reads img from the path specified and returns only the sudoku puzzle
    """
    print("extracting the sudoku grid.....")
    
    # read the image from path
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    
    contours, _  = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # find the ten contours having the largest area
    contours = sorted(contours,key=cv2.contourArea,reverse=True)[:10]

    sudoku_square = None
    for c in contours[1:]:
        # approximate the contour to a polygon (closed figure)
        # for more info refer: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)

        # if the polygon has four sides break (largest 4 sided polygon)
        if len(approx) == 4:
            sudoku_square = approx
            break
    
    # return the image after performing perspective transform
    # refer more here: https://docs.opencv.org/master/da/d6e/tutorial_py_geometric_transformations.html
    return(grab_warp(img,sudoku_square))

"""
    For perspective transformation, you need a 3x3 transformation matrix. Straight lines will remain straight even after the transformation. To find this transformation matrix, you need 4 points on the input image and corresponding points on the output image. Among these 4 points, 3 of them should not be collinear. Then the transformation matrix can be found by the function cv.getPerspectiveTransform. Then apply cv.warpPerspective with this 3x3 transformation matrix.
"""

def grab_warp(img,sudoku_square):
    """ performs perspective transform """

    # converting the cordinates of the sudoku square into required format (4 corners of the sudoku square) 
    pts = sudoku_square.reshape((4,2))
    rect = get_rect(pts)

    (maxHeight,maxWidth) = getMax_HW(rect)

    # 4 corners of the final sudoku square
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    

    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warp

def get_rect(pts):
    """
        helper function to find the four corners of the 
        sudoku square in original image
    """
    rect = np.zeros((4,2),dtype=np.float32)
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def getMax_HW(rect):
    """
        helper function to find the maximum width and 
        height of the sudoku square in original image
    """
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    return (maxHeight,maxWidth)