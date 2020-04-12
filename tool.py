import Model.model as md
import Utils.find_digit as fd
import Utils.solve as s
import Preprocess.preprocess as pp
from torchvision import transforms
import argparse
import cv2
import numpy as np
import torch

def find_numbers(cells_raw):
    cells = []
    for cell in cells_raw:
        cell = fd.get_digit(cell)
        # resize = cv2.resize(cell, (32,32), interpolation = cv2.INTER_AREA)
        resize = cv2.resize(cell, (48,48), interpolation = cv2.INTER_AREA)
        cells.append(resize)
    return cells

def digitalize(model,cells):
    print("digitalizing the given image.....")
    sudoku_grid = []
    row = []
    transform = transforms.Compose([transforms.ToTensor()])
    k = 0
    for cell in cells:
        cell = transform(cell)
        # predictions = model(cell.view(1,1,32,32))
        predictions = model(cell.view(1,1,48,48))
        value = int(predictions.argmax())
        row.append(value)
        k += 1
        if(k % 9 == 0):
            sudoku_grid.append(row)
            row = []
    return sudoku_grid

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", help="path to the image of the sudoku problem",required=True)
    args = vars(parser.parse_args())
    path = args["image_path"]

    cells_raw = pp.preprocess(path)
    cells = find_numbers(cells_raw)
    model = md.load_model()
    sudoku_grid = digitalize(model,cells)
    
    print("displaying the sudoku grid.....")
    for row in sudoku_grid:
        for cell in row:
            print(cell,end=" ")
        print()
    print("solving the puzzle")
    solved_grid = s.solve(sudoku_grid)
    print("displaying the solution.....")
    for row in solved_grid:
        for cell in row:
            print(cell,end=" ")
        print()