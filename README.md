# Sudoku-solver
sudoku solver which takes an image as input and displays the solved output in the terminal

## Methadology Used:
1) Read the input image and find the puzzle grid inside the image
![extracting the sudoku puzzle from the image](https://github.com/saoalo/Sudoku-solver/blob/master/screen_shots/extract_grid.png)


2) Break the puzzle (9x9 grid) into 81 cells (divide the image into 81 equal size squares) 
![extracting the sudoku puzzle from the image](https://github.com/saoalo/Sudoku-solver/blob/master/screen_shots/break_into_cells.png)


3) The cells have remnants of the grids border. These border lines will act as noise when we are trying to identify the number ,hence
we need to extract only the digit from the cell
![extracting the sudoku puzzle from the image](https://github.com/saoalo/Sudoku-solver/blob/master/screen_shots/extract_digitpng.png)


4) Now we pass the images to a convolutional neural network to digitalize the numbers, and then solve the puzzle (using a parallel algorithms
[tutorial](https://cse.buffalo.edu/faculty/miller/Courses/CSE633/Sankar-Spring-2014-CSE633.pdf) [github repo](https://github.com/Shivanshu-Gupta/Parallel-Sudoku-Solver))
![final result](https://github.com/saoalo/Sudoku-solver/blob/master/screen_shots/final.png)

## Packages Used:
```
1 opencv==4.1.0
2 pytorch==1.1.0
3 numpy==1.16.4
4 torchvision==0.3.0
5)g++ (Ubuntu 9.2.1-9ubuntu2) 9.2.1 20191008
6)openmp
```
