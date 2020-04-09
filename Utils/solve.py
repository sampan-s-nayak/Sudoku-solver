import subprocess

def solve(grid,solver_path='./Solver/./a.out',num_threads=4):
    """
        passes the digitized sudoku puzzle to the solver and return the final answer
            grid: the sudoku grid(9x9) in numeric format (int array)
            solver_path: the path to the solver being used
            num_threads: number of threads to be created by the solver to solve the puzzle
            (by default a parallel solver is used) 
    """
    # converting the grid into a string
    grid_str = '\n'.join([str(i) for j in grid for i in j])

    # sending the string to the solver and reading the output
    proc = subprocess.run(solver_path,stdout=subprocess.PIPE,text=True,input=grid_str)
    data = proc.stdout

    # converting the output back into an array of numbers
    grid = [[0 for x in range(9)] for y in range(9)]
    nums = [float(i) for i in data.split() if i != '\n']
    for i in range(9):
        for j in range(9):
            grid[i][j] = int(nums[i*9+j])
    time_taken = nums[81]*1000
    print("solved in " + str(time_taken)+'ms')

    # returning the solved puzzle
    return grid