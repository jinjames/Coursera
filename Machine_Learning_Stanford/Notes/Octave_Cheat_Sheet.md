# Octave Cheat Sheet

## Basic Operations
1. **Not equals:** "~=" or "!="
2. **Logical and:** "&&"
3. **Logical or:** "||"
4. **Round down:** "floor(a)"
5. **Round up:** "ceil(a)"

## Vectors and Matrices

### Creation and Assignment
1. **Standard assignment**
    To create a 3x2 matrix, A:
    ```
    A = [1 2; 3 4; 5 6]
    ```

2. **Range vectors**
    To create a row vector, y, with elements from 1 to 2 in increments of 0.1:
    ```
    y = 1:0.1:2
    ```

3. **Populate a matrix with one value**
    To create a 2x3 matrix, B, with all 2's:
    ```
    B = 2 * ones(2,3)
    ```

4. **Generate a matrix with random values**
    To create a 1x3 matrix, C, with random values between 0 and 1:
    ```
    rand(1,3)
    ```
    To create a 1x3 matrix, D with random values from a Gaussian distribution with mean 0 and variance 1:
    ```
    randn(1,3)
    ```

5. **Generate the identity matrix**
    To create the 3x3 identity matrix:
    ```
    eye(3)
    ```

6. **Assign values to one row**
    To assign the second columnof matrix X to a specified vector:
    ```
    X(:, 2) = [10; 11; 12]
    ```

### Access
1. **Specific element**
    To access matrix X's element in the 3rd row, 2cnd column:
    ```
    X(3,2)
    ```

2. **Range of a matrix**
    To access the first ten elements of matrix Y:
    ```
    Y(1:10)
    ```

3. **All elements in a row or column**
    To access all elements in the 2$^{nd}$ row of matrix Z:
    ```
    Z(2,:)
    ```
    To access all elements in the 3$^{rd}$ column of matrix Z:
    ```
    Z(:,3)
    ```
    To access all elements in the $1^{st}$ and $3^{rd}$ row of matrix Z:
    ```
    Z([1 3], :)
    ```

### Operations
1. **Append a column to matrix A:**
    ```
    A = [A, [100; 101; 102]]
    ```

2. **Serialize all elements into a vector:**
    ```
    A(:)
    ```

3. **Concatenate two matrices**
    Concatenate matrix A with B, in order, side by side:
    ```
    C = [A B]
    ```
    Concatenate matrix A with B, in order, with A on top and B on the bottom
    ```
    C = [A; B]
    ```

4. **Multiplication**
    Matrix multiplication of matrices A and B:
    ```
    C = A * B
    ```
    Scalar multiplication of matrices A and B:
    ```
    C = A .* B
    ```

5. **Element-wise operations:**
    Typically, just add a "." before the operator.
    For example, to reciprocate all elements in a matrix, V:
    ```
    V = 1 ./ V
    ```

6. **Add a value to each element in a matrix**
    To increment each element in matrix A by 1:
    ```
    A = A + 1
    ```

7. **Transpose**
    To get matrix A's transpose:
    ```
    A'
    ```

8. **Check if each element in a matrix satisfies a condition**
    Check if each element in a matrix, A, is less than 3:
    ```
    A < 3
    ```
    Check if each element in a matrix, A, is less than 3 and return two vectors for the corresponding columns and rows of the elements:
    ```
    [row, column] = A < 3
    ```
    Find the indices of the elements that satisfy the condition:
    ```
    find(A < 3)
    ```

9. **Run an operation on all elements of a matrix consecutively**
    Sum all elements of a matrix, A:
    ```
    sum(A)
    ```
    Multiply all elements of a matrix, A:
    ```
    prod(A)
    ```

10. **Maximum and minimum values in a matrix**
    Get the maximum of each column of a matrix, A:
    ```
    max(A, [], 1)
    ```
    Get the minimum of each row, of a matrix, A:
    ```
    min(A, [], 2)
    ```
    Absolute max/min of a matrix, A:
    ```
    max(max(A))
    max(A(:))
    ```

11. **Maximum and minimum values in a vector**
    To get the maximum/minimum values and their indices in the vector a:
    ```
    [value, index] = max(a)
    [value, index] = min(a)
    ```

12. **Sum of all diagonal elements in a 9x9 matrix, A:**
    ```
    sum(sum(A .* eye(9)))
    ```

13. **Reflect a matrix, A, over the Y-axis:**
    ```
    flipud(A)
    ```

14. **Compute the pseudo-inverse of a matrix, A:**
    ```
    pinv(A)
    ```

15. **Take the log for each element in a matrix, A:**
    ```
    log(A)
    ```

#### Other
1. **Dimensions of a matrix:**
    To get the size of a matrix, A:
    ```
    size(A)
    ```
    To get the largest dimension of a matrix, A:
    ```
    length(A)
    ```

## Plotting
1. **Generate a histogram from a matrix, W, with 50 bins:**
    ```
    hist(W, 50)
    ```

2. **Plot vectors x vs. y1:**
    ```
    plot(x, y1)
    ```

3. **Plot a second vector, y2, in red:**
    ```
    hold on
    plot(x, y2, 'r')
    ```

4. **Axes**
    Set the x-axis label:
    ```
    xlabel('time')
    ```
    Set the y-axis label:
    ```
    ylabel('value')
    ```
    Set the x-axis to range from 0 to 1 and the y-axis to range from -1 to 1:
    ```
    axis([0 1 -1 1])
    ```

5. **Legend**
    Set the legend label for y1 to be 'sin' and y2 to be 'cos':
    ```
    legend('sin', 'cos')
    ```

6. **Title**
    Set the title for the figure:
    ```
    title("My Plot")
    ```

7. **Save a plot:**
    ```
    print -dpng "myPlot.png"
    ```

8. **Create multiple figures**
    Create the first figure (x vs. y1) and then the second (x vs. y2):
    ```
    figure(1); plot(x, y1)
    figure(2); plot(x, y2)
    ```

9. **Subplots**
    Create a 1 by 2 grid of plots, and select the first subplot:
    ```
    subplot(1, 2, 1)
    ```

9. **Clear the figure:**
    ```
    clf
    ```

10. **Color bars**
    Create a color bar plot of matrix A in greyscale, with a bar on the right indicating what value each shade corresponds to:
    ```
    imagesc(A), colorbar, colormap gray
    ```

## Interfacing with Data
1. **Load data from a file:**
    ```
    load('fileName.ext')
    ```

2. **Save a variable to file**
    In binary format:
    ```
    save fileName.ext variableName
    ```
    In human readable format:
    ```
    save fileName.ext variableName -ascii
    ```

3. **Print variables in use:**
    ```
    whos
    ```

## Control Statements

1. **For loops:**
    ```
    for i = 1:10,
      doThisCode()
    end
    ```

2. **While loops:**
    ```
    while i <= 5,
      doThisCode()
    end
    ```
3. **Break statements:**
    ```
    while true,
      doThisCode()
      if true,
        break
      end
    end
    ```

4. **If statements:**
    ```
    if x==1,
      doThisCode()
    elseif x==2,
      doThisOtherCode()
    else,
      doThatCode()
    end
    ```

5. **Existence of a value**
    Determine if the value 2 exists in the matrix A:
    ```
    any(A==2)
    ```

## Functions
1. **Defining a function**
    Define a function called "bar", which squares and cubes its argument "x" and returns the results as "first" and "second":
    ```
    function [first, second] = bar(x)
      first = x^2
      second = x^3
    ```

2. **Wrapping a function**
    Create a function with argument t, which calls another function "myFunc":
    ```
    @(t)(myFunc(t, X, y))
    ```

## Miscellaneous
1. Comments
    ```
    % This is a comment
    ```
2. Change the prompt symbol:
    ```
    PS1('>> ')
    ```
3. Suppress output:
    ```
    A(3,2); % The ";" suppresses output
    ```
4. Change accuracy of prints:
    ```
    format long % High accuracy
    format short % Low accuracy
    ```
5. Add a directory to Octave's search path:
    ```
    addpath("C:\search\path")
    ```

## Optimization Algorithms
1. **fminunc**
    - Pass to the algorithm: a pointer to the cost function, the vector for the initial guess of theta, and "options", which provides parameters including the max number of iterations.
    - **exitFlag = 1 if the algorithm converged, otherwise exitFlag = 0**

    ```
    options = optimset('GradObj', 'on', 'MaxIter', '100');
    [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options)
    ```
