# Coursera: Machine Learning at Stanford

#### Notes by: James Jin

# Week 1

## Introduction

### Definitions
**Machine learning:** A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.

### Types of Learning

#### Supervised Learning
- **Given the "right answer"** for each example in the data
- **Uses feedback** from the output to improve the prediction algorithm

##### Types of Supervised Learning
1. Regression
    - Predict **real-valued, continuous** output
    - e.g. Predicting the price of a house based on its size

2. Classification
    - Predict **discrete** output
    - e.g. Determining if a tumor is benign or malignant

#### Unsupervised Learning
- Derive structure from data where the **relationship between the input and output is unknown**
- **No feedback** based on the prediction results
- e.g. "The Cocktail Party Algorithm": Separate individual voices from a mesh of  sounds

## Model and Cost Function

``` mermaid
graph LR
  Input(Input) --> Hypothesis(Hypothesis, h)
  Hypothesis --> Output(Output)
```

### Notation
1. **x**
    - Input variable, called **features**

2. **y**
    - Output variable, called **target**

3. **m**
    - Number of training examples

4. **$(x^{(i)}, y^{(i)})$**
    - i^th^ training example
5. **$\theta_{i}$**
    - i^th^ input's coefficient, called a **parameter**
6. **=**
    - Truth **assertion**
    - e.g. a = b
7. **:=**
    - Value **assignment**
    - e.g. a := a + 1

### Single Variable (Univariate) Linear Regression

#### Hypothesis
- Guess the prediction output based on inputs

***
``` math
h_{\theta}(x) = \theta_{0} + \theta_{1}(x)
```
***

#### Cost Function

##### Squared Error Function
- **In the equation below:** The $\frac{1}{2}$ is used for convenience, since the derivative term will cancel it out

***
``` math
J(\theta_{0}, \theta_{1}) = \frac{1}{2m} \sum_{i=1}^{m}(h_{\theta}(x_{i}) - y_{i})^2
```
***

#### Gradient Descent

##### Algorithm
***
repeat until convergence {
``` math
\theta_{j} := \theta_{j} - \alpha\frac{\partial}{\partial\theta_{j}}J(\theta_{0}, \theta_{1}),\; (for\, j=0\, and\, j=1)
```
}

**or**

repeat until convergence {
``` math
\theta_{j} := \theta_{j} - \alpha \frac{1}{m} \sum_{i=1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)}) * x^{(i)}
```
}
***

##### Updating Theta
- Each value of theta must be updated together **simultaneously** (i.e. calculate values, then update)
``` math
temp0 := \theta_{0} - \alpha \frac{\partial}{\partial\theta_{0}}J(\theta_{0}, \theta_{1})
```
``` math
temp1 := \theta_{1} - \alpha \frac{\partial}{\partial\theta_{1}}J(\theta_{0}, \theta_{1})
```
``` math
\theta_{0} := temp0
```
``` math
\theta_{1} := temp1
```

#### Learning Rate, $\alpha$
- Value can be fixed during gradient descent - gradient descent will automatically take "smaller steps"

##### If $\alpha$ is too small
- Gradient descent can be **slow**

##### If $\alpha$ is too large
- Gradient descent can overshoot the minimum (**fail to converge**, or even diverge)

#### Batch Gradient Descent
- Each step of gradient descent uses all the training examples

## Linear Algebra Review

### Matrices and Vectors

#### Notation and Definitions
1. **$A_{i,j}$**
    - Refers to the element in the i^th^ row and j^th^ column of a matrix, $A$

2. **$a_{i}$**
    - Refers to the element in the i^th^ row of a vector, $a$
3. **Indexing**
    - Assume **1-indexing** unless otherwise specified
4. **Dimension**
    - Number of rows * number of columns

#### Addition and subtraction
- Element-wise
- Operands' dimensions must be the same

#### Scalar multiplication and division
- Perform multiplication or division on every element by the scalar value

#### Matrix multiplication

##### Properties
1. **Not Commutative**
    ``` math
    A \times B \neq B \times A
    ```

2. **Associative**
    ``` math
    A \times (B \times C) = (A \times B) \times C
    ```

3. **Identity Matrix, $I_{nxn}$**
    e.g. $I_{3x3}$:
    ``` math
    \begin{matrix}
        1 & 0 & 0 \\
        0 & 1 & 0 \\
        0 & 0 & 1
    \end{matrix}
    ```

    **For any $I_{nxn}$:**
    ``` math
    A \times I = I \times A = A
    ```

##### Dimension Requirements
- Multiplying two matrices, $A_{m1xn1}$ and $B_{n2xm2}$ produces an output matrix $C_{m1xm2}$
- In the example above, we **must** have:
    - m1 = m2 and n1 = n2

##### Process
1. Multiply $A$'s i^th^ row with $B$'s j^th^ column
2. Sum up all values to get $C_{i,j}$

#### Inverses
- **Not all matrices have an inverse**
- If the inverse exists:
    1. The matrix must be square
    2. The following condition must be true:
    ``` math
    A(A^{-1}) = A^{-1}(A) = I
    ```

#### Transposes
- $B_{nxm}$ is the transpose of $A_{mxn}$ (i.e. $B$ = $A^T$) if:

``` math
B_{i,j} = A_{j,i}
```

# Week 2

## Multivariate Linear Regression

### Notation
1. **n**
    - Number of features ($x_{1}$, $x_{2}$, ...)
2. **$x_{j}^{(i)}$**
    - Value of feature $j$ in i^th^ training example

### Hypothesis
***
``` math
h_{\theta}(x) = \theta_{0} + \theta_{1}x_{1} + ... + \theta_{n}x_{n}
```
**or**
``` math
h_{\theta}(x) = \theta^{T}x
```
where $\theta^T$ is a 1 x (n+1) matrix
***

### Cost Function

#### Squared-Error Function
***
``` math
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)})^2
```
***

### Gradient Descent

#### Algorithm
***
repeat until convergence: {
``` math
\theta_{j} := \theta_{j} - \alpha \frac{1}{m} \sum_{i=1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)}) \times x_{j}^{(i)},\; for\, j = 0\, to\, n
```
}
***

#### Verification
1.  $J(\theta)$ should **always decrease** after every iteration
    - If this is not the case: **use a smaller $\alpha$**

### Feature Scaling and Mean Normalization

#### Goal
- Ensure that **each feature is on a similar scale**
- Allows gradient descent to **converge more quickly**
- **Every feature should satisfy: $-1 \leq x_{i} \leq 1$**

#### Techniques

##### Feature Scaling
- Divide input values by the range (i.e. max value - min value) of the input variable

##### Mean normalization
- Subtract the mean value from the input values  

##### Overall
- To implement feature scaling and mean normalization together:
***
``` math
x_{i} := \frac{x_{i} - \mu_{i}}{s_{i}}
```
***

## Computing Parameters Analytically

### Normal Equation
- Analytically computes the $\theta$ that minimizes $J$, without need for iteration
- Do **not** use feature scaling with the normal equation

***
``` math
\theta = (X^{T}X)^{-1} X^{T}y
```
***

#### Gradient Descent vs. Normal Equation

| Gradient Descent | Normal Equation |
| :---: | :---: |
| Need to select alpha | No need to choose alpha |
Needs many iterations | Computed analytically |
O(kn^2^) | O(n^3^) (need to calculate inverse of X^T^X) |
Works well when n is large | Slow if n is large |

### Noninvertibility

#### Causes
1. **Redundant features (linearly dependent)**
    - e.g. If $x_{1}$ is size in meters, while $x_{2}$ is size in feet
    - Can delete redundant features

2. **Too many features**
    - e.g. $m \leq n$
    - Can eliminate insignificant features, or use regularization

# Week 3

## Classification and Representation

### Classes
- In binary classification, there are two classes:
    1. **Negative class** $\rightarrow$ "0"
    2. **Positive class** $\rightarrow$ "1"

### Classification Constraints
- **For Linear Regression:** $h_{\theta}(x)\; can\; be >1\; or <0$
- **For Logistic Regression:** $0 \leq h_{\theta}(x) \leq 1$

### Threshold Classification
- Assume Z = 0.5 (can be any value):
  1. If $h_{\theta}(x)$ >= Z, predict y = 1
  2. If $h_{\theta}(x)$ < Z, predict y = 0
- **Very ineffective if there are outliers**

### Logistic Regression

#### Hypothesis
- g(z) is the **logistic** or **sigmoid** function
***
``` math
g(z) = \frac{1}{1 + e^{-z}}
```
``` math
h_{\theta}(x) = g(\theta^Tx) = \frac{1}{1 + e^{-\theta^{T}x}}
```
***
**Predict y = 1:**
``` math
h_{\theta}(x) \geq 0.5
```
``` math
\theta^{T}x \geq 0
```
**Predict y = 0:**
``` math
h_{\theta}(x) \lt 0.5
```
``` math
\theta^{T}x \lt 0
```
***

#### Decision Boundary
- Contour that separates the regions where the hypothesis predicts y = 0 and y = 1
- It is a property of the hypothesis, **not** of the data set

##### Calculations
1. **The boundary is defined at:**
Z = $\theta_{0} + \theta_{1}x_{1} + \theta_{2}x_{2} + ... = 0$

2. **y = 0 when:**
Z = $\theta_{0} + \theta_{1}x_{1} + \theta_{2}x_{2} + ... \lt 0$

3. **y = 1 when:**
Z = $\theta_{0} + \theta_{1}x_{1} + \theta_{2}x_{2} + ... + \geq 0$

#### Cost Function
***
``` math
J(\theta) = \frac{1}{m} \sum_{i = 1}^{m} Cost(h_{\theta}(x^{(i)}), y^{(i)})
```
``` math
J(\theta) = -\frac{1}{m} [\sum_{i = 1}^{m} y^{(i)}logh_{\theta}(x^{(i)}) + (1 - y^{(i)})log(1 - h_{\theta}(x^{(i)}))]
```
**Vectorized:**
``` math
J(\theta) = \frac{1}{m} (-y^{T}log(h) - (1-y)^{T}log(1-h))
```
***
``` math
Cost(h_{\theta}(x), y) =  \begin{cases}
      -log(h_{\theta}(x)) & if\, y = 1 \\
      -log(1-h_{\theta}(x)) & if\, y = 0
   \end{cases}
```
``` math
Cost(h_{\theta}(x), y) = -ylog(h_{\theta}(x)) - (1-y)log(1-h_{\theta}(x))
```
***

**If y = 0:**
``` math
If\; h_{\theta}(x) = 0: Cost = 0
```
``` math
If\; h_{\theta}(x) \rightarrow 1: Cost \rightarrow \infty
```
**If y = 1:**
``` math
If\; h_{\theta}(x) = 1: Cost = 0
```
``` math
If\; h_{\theta}(x) \rightarrow 0: Cost \rightarrow \infty
```

##### Convex Functions
- The same global maximum can be found regardless of where gradient descent is started

#### Gradient Descent
- Cosmetically, equation looks identical to Linear Regression Gradient Descent - the difference is in the hypothesis
***
Repeat {
``` math
\theta_{j} := \theta_{j} - \alpha \sum_{i=1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)}) \times x_{j}^{(i)}
```
}

**Vectorized:**
``` math
\theta := \theta - \frac{\alpha}{m}X^{T}(g(X\theta) - \vec{y})
```
***

## Multiclass Classification

## Solving the Problem of Overfitting
