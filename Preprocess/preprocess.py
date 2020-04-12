from Preprocess import find_grid
from Preprocess import grid_extractor

def preprocess(img_path):
    """
        helper function to
        1) find the sudoku puzzle and extract it
        2) extract the individual cells from the sudoku grid
    """

    # extract the sudoku grid from the given image
    sudoku_grid = find_grid.extract_grid(img_path)

    # extract the cells from the sudoku grid
    grid_cells = grid_extractor.extract_cells(sudoku_grid)

    # return the extracted cells
    return grid_cells