# Sudoku-solver
sudoku solver which takes an image as input and displays the solved output in the terminal. A neural network trained in Pytorch is used to identify the numbers present in the sudoku puzzle

## Running The Program

```
python3 tool.py -i <image_path>

```

## Neural Network Architecture

two Networks have been tested
<br />
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(5*5*64, 500)
        self.fc2 = nn.Linear(500, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```    
this network works with the weights named checkpoint1.pth to checkpoint5.pth
<br /><br />

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, stride=1)
        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.norm2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(9*9*64, 500)
        self.dropout1 = nn.Dropout(p=0.8)
        self.fc2 = nn.Linear(500, 10)
    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        # flattening the image
        x = x.view(-1, 9*9*64)
        x = self.dropout1(F.relu(self.fc1(x)))
        # final output
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```
this network with the weights named checkpoint_new1.pth. this network is used by default
<br />
The Weights are stored in the Weights folder 
<br /><br />
## Dataset Used

I found the dataset used in this project in some github repository. I dont have a link to that github repository so I have stored a copy of the dataset in this repository. Once I find the original source, I will update this reference section and remove the dataset from this repository.  
<br /><br />

## Methadology Used:
1. Read the input image and find the puzzle grid inside the image
![extracting the sudoku puzzle from the image](https://github.com/saoalo/Sudoku-solver/blob/master/screen_shots/extract_grid.png)  
<br /><br />

2. Break the puzzle (9x9 grid) into 81 cells (divide the image into 81 equal size squares) 
![extracting the sudoku puzzle from the image](https://github.com/saoalo/Sudoku-solver/blob/master/screen_shots/break_into_cells.png)  
<br /><br />

3. The cells have remnants of the grids border. These border lines will act as noise when we are trying to identify the number ,hence
we need to extract only the digit from the cell
![extracting the sudoku puzzle from the image](https://github.com/saoalo/Sudoku-solver/blob/master/screen_shots/extract_digitpng.png)
<br /><br />

4. Now we pass the images to a convolutional neural network to digitalize the numbers, and then solve the puzzle (using a parallel algorithms
[tutorial](https://cse.buffalo.edu/faculty/miller/Courses/CSE633/Sankar-Spring-2014-CSE633.pdf) [github repo](https://github.com/Shivanshu-Gupta/Parallel-Sudoku-Solver))
![final result](https://github.com/saoalo/Sudoku-solver/blob/master/screen_shots/final.png)  
<br /><br />

## Requirements:
```
1 opencv
2 pytorch
3 numpy
4 torchvision
5)g++ compiler
6)openmp
```
<br /><br />
### Todo
1. Add an option to use a serial sudoku solver written in python instead of the parallel algorithm written in c++(using openmp) being used right now
2. find a better way to extract the sudoku grid.
3. find a better way to extract the digits from the cell
4. Improve the neural network (observation: resizing the digit images to (28x28) seemed to work better than resizing it to (48x48) (this is being used in the current implementation) )
5. finding more data to train the model
