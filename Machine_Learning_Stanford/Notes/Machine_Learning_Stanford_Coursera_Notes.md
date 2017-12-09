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

#### Addition and Subtraction
- Element-wise
- Operands' dimensions must be the same

#### Scalar Multiplication and Division
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
    \begin{bmatrix}
        1 & 0 & 0 \\
        0 & 1 & 0 \\
        0 & 0 & 1
    \end{bmatrix}
    ```

    **For any $I_{nxn}$:**
    ``` math
    A \times I = I \times A = A
    ```

4. **Multiplication by transpose**
    - If $a$ and $b$ are both vectors:
    ``` math
    a^{T}b = b^{T}a
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

**Note:** Logistic regression is a **linear qualifier** - it can only determine linear decision boundaries!

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

#### Other Optimization Algorithms vs. Gradient Descent

##### Algorithms
1. Conjugate Gradient
2. BFGS
3. L-BFGS

##### Comparison
| Advantages | Disadvantages |
| :---: | :---: |
| - Don't need to pick $\alpha$ <br> - Typically faster than gradient descent | - More complex |

## Multiclass Classification
- Value of y can be from any set of **discrete** values

### One vs. All Algorithm
- Train a logistic regression classifier $h_{\theta}^{(i)}(x)$ **for each class $i$** to predict the probability that $y = i$

***
``` math
h_{\theta}^{(i)}(x) = P(y = i|x;\theta)\;\;\;\;\; (i = 1, 2, 3)
```
***

- To make a prediction on a new input $x$, pick class $i$ that maximizes $h_{\theta}^{(i)}(x)$

## Solving the Problem of Overfitting

### Underfit (High Bias)
- Learned hypothesis does not fit the training set well, and thus also fails to fit new examples

### Overfit (High Variance)
- Learned hypothesis may fit the training set very well, but fails to generalize to new examples

#### Addressing Overfitting
1. Reduce number of features
2. Regularization
    - Useful when each parameter, $\theta_{j}$ has a small impact on the prediction

### Regularization
- Keep all features, but reduce magnitude of parameters, $\theta_{j}$
- Creates a simpler hypothesis to address overfitting

***
``` math
J(\theta) = \frac{1}{2m}[\sum_{i=1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{i=1}^{n}\theta_{j}^2]
```
where $\lambda$ is the **regularization parameter**
***

- **If $\lambda$ is too large, there can be underfitting**, since each $\theta_{j} \rightarrow 0$

### Regularized Linear Regression

#### Gradient Descent
**Note:** Below, $1 - \alpha \frac{\lambda}{m} \lt 1$

***
Repeat {
``` math
\theta_{0} := \theta_{0} - \alpha\frac{1}{m} \sum_{i=1}^{m}(h_{\theta}(x^{(i)}) - y^{((i))}) \times x_{0}^{(i)}
```
``` math
\theta_{j} := \theta_{j}(1 - \alpha \frac{\lambda}{m}) - \alpha \frac{1}{m} \sum_{i=1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)}) \times x_{j}^{(i)}
```
for j = 1, 2,...,n
}
***

#### Normal Equation
***
``` math
\theta = (X^{T}X + \lambda(M))^{-1} X^{T}y
```
where **M is a (n+1, n+1) matrix that has 1s on its diagonal, expect for the first index**

e.g. If n = 2:

 M = $\begin{bmatrix}
    0 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1
\end{bmatrix}$
***

#### Non-Invertibility
- Without regularization, if $m \leq n$, $X^{T}X$ is non-invertible and we cannot find $\theta$ using the normal equation

- With regularization, if $m \leq n$ **and $\lambda \gt 0$**, $X^{T}X + \lambda(M)$ is still invertible and the normal equation can be used

#### Cost Function
***
``` math
J(\theta )= -[\frac{1}{m} \sum_{i=1}^{m} y^{(i)}logh_{\theta}(x^{(i)}) + (1-y^{(i)})log(1-h_{\theta}(x^{(i)}))] + \frac{\lambda}{2m} \sum_{j=1}^{n}\theta_{j}^2
```
for $\theta_{1}, \theta_{2}, ... , \theta{n}$
***

#### Gradients
- For use in implementing the optimization algorithn in Octave

***
``` math
\frac{\partial}{\partial\theta_{j}}J(\theta) = \frac{1}{m} \sum_{i=1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)}) \times x_{j}^{(i)} + \frac{\lambda}{m}\theta_{j}
```
***

### Points of Interest
1. Adding a new feature **always** results in **equal or better** performance on the **training set**

# Week 4

## Motivations
- Neural networks come into use when we need to fit a difficult data set with a hypothesis, but standard methods (e.g. multivariate linear regression) would require too many features
    - i.e. With n features, **including just the quadratic terms (e.g. $x_i$$x_j$) would produce roughly $\frac{n^2}{2}$ features**

## Neural Networks

### Neurons in the Brain
- Dendrite $\rightarrow$ "input wire"
- Axon $\rightarrow$ "output wire"

### Neuron Model: A Logistic Unit

``` mermaid
graph LR

x0((x0)) --> node(( ))
x1((x1)) --> node(( ))
x2((x2)) --> node(( ))
x3((x3)) --> node(( ))
node(( )) --> hyp(("h(x)"))
```

- The $x_0$ term is called the **Bias Neuron/Unit** and is always 1

### Terminology
1. **Sigmoid (Logistic) Activation Function**
    - Uses the logistic function

2. **Weights**
    - Parameters ($\theta$)

3. **Input Layer**
    - The first layer

4. **Output Layer**
    - The last layer

5. **Hidden Layer**
    - Any layer that is not the input or output

### Notation
1. **$a_{i}^{(j)}$**
    - "Activation" of unit $i$ in layer $j$

2. **$\theta^{(j)}$**
    - Matrix of weights controlling function mapping from layer $j$ to layer $j+1$

### Computing the Hypothesis: Forward Propagation

#### Example
``` mermaid
graph LR

x1((x1)) --> a1(("a1(2)"))
x1 --> a2
x1 --> a3
x2((x2)) --> a2(("a2(2)"))
x2 --> a1
x2 --> a3
x3((x3)) --> a3(("a3(2)"))
x3 --> a1
x3 --> a2
a1 --> out((" "))
a2 --> out
a3 --> out
out --> hyp(("h(x)"))
```

- Contains:
    1. **Layer 1: Input layer**
        - 3 units
    2. **Layer 2: Hidden layer**
        - 3 hidden units
    3. **Layer 3: Output layer**
        - 1 unit

***
- The **z values** are (in vectorized form):
``` math
z^{(2)} = \theta^{(1)} \times a^{(1)}
```
``` math
z^{(3)} = \theta^{(2)} \times a^{(2)}
```
***
- The **activation values** are:
``` math
a^{(l)} = g(\theta^{(l-1)} \times a^{(l-1)})
```
``` math
a_{1}^{(2)} = g(z_{1}^{(2)}) = g(\theta_{10}^{(1)}x_{0} + \theta_{11}^{(1)}x_{1} + \theta_{12}^{(1)}x_{2} + \theta_{13}^{(1)}x_{3})
```
``` math
a_{2}^{(2)} = g(z_{2}^{(2)}) = g(\theta_{20}^{(1)}x_{0} + \theta_{21}^{(1)}x_{1} + \theta_{22}^{(1)}x_{2} + \theta_{23}^{(1)}x_{3})
```
``` math
a_{3}^{(2)} = g(z_{3}^{(2)}) = g(\theta_{30}^{(1)}x_{0} + \theta_{31}^{(1)}x_{1} + \theta_{32}^{(1)}x_{2} + \theta_{33}^{(1)}x_{3})
```
- **Note:** $a^{(2)}$ should be 4 terms long, since there is also a $a_{0}^{(2)}$, which is not shown here
***
- The **hypothesis value** is:
``` math
h_{\theta}(x) = a_{1}^{(3)} = g(z^{(3)}) =  g(\theta_{10}^{(2)}a_{0}^{(2)} + \theta_{11}^{(2)}a_{1}^{(2)} + \theta_{12}^{(2)}a_{2}^{(2)} + \theta_{13}^{(2)}a_{3}^{(2)})
```
***

### Points of Interest
- **Note:** If a network has $s_{j}$ units in layer $j$, $s_{j+1}$ units in layer $j + 1$, then $\theta^{(j)}$ will have dimension $(s_{j+1}) \times (s_{j} + 1)$

## Applications

### Hypothesis Functionality
- Can determine the functionality of a hypothesis by creating its truth table

#### Process
This example is for logical NOR:
1. **Write out hypothesis**
    $h_{\theta}(x) = g(10 - 20x_{1})$

2. **Calculate hypothesis value for the different possible permutations of the features**
    | $x_{1}$ | $h_{\theta}(x)$ |
    | :---: | :---: |
    | 0 | g(10) = 1 |
    | 1 | g(-10) = 0 |

### Multiclass Classification Neural Networks

#### Output Neurons
- With $n$ possible outputs, there must be **$n$ output neurons** (one for each possible output)

#### Output Vector, $y$
- $y$ will be a **column vector of size $n$**, where **one element has a value of one** and the **rest are zero**
e.g. With 3 possible outputs, y will be one of:
``` math
\begin{bmatrix}
    1 \\
    0 \\
    0
\end{bmatrix}

\begin{bmatrix}
    0 \\
    1 \\
    0
\end{bmatrix}

\begin{bmatrix}
    0 \\
    0 \\
    1
\end{bmatrix}
```

#### Hypothesis, $h_{\theta}(x)$
- Ideally our hypothesis would be the same as $y$, in that it would be a **column vector of size $n$**, where **one element has a value of one** and the **rest are zero**

# Week 5

## Cost Function and Back Propagation

### Terminology
1. **$L$**
    - Total number of layers in network

2. **$s_{l}$**
    - Number of units (not counting the bias unit) in layer $l$

3. **$K$**
    - Number of units in the output layer

### Cost Function
- Note that in the equation below, the indices **start at 1, not 0**, since we do not want to consider the bias unit
***
``` math
J(\theta) = -\frac{1}{m} [\sum_{i=1}^{m} \sum_{k=1}^{K}y_{k}^{(i)} log(h_{\theta}(x^{(i)}))_{k} + (1-y_{k}^{(i)})log(1-(h_{\theta}(x^{(i)}))_{k})] + \frac{\lambda}{2m} \sum_{l}^{L-1} \sum_{i=1}^{s_{l}} \sum_{j=1}^{s_{l+1}} (\theta_{ji}^{l})^{2}
```
***

### Gradient Computation: Backpropagation
- An algorithm to help minimize the cost function by computing the gradients

#### Single Example

1. **Delta, $\delta_{j}^{(l)}$**

    ##### Definition
    - Informally: "Error" of node $j$ in layer $l$
    - Formally: $\delta_{j}^{(l)} = \frac{\partial}{\partial z_{j}^{(l)}}cost(i)$, for $j \geq 0$,
      where cost(i) = $y^{(i)}logh_{\theta}(x^{(i)}) + (1-y^{(i)})logh_{\theta}(x^{(i)})$

    ##### Points of Note
    - $\delta^{(1)}$ **does not exist**
    - For any $l$, $\delta_{0}^{(l)}$ = 1. **Ignore and do not use these values**

    - For each **output** layer:
    ***
    ``` math
    \delta_{j}^{(l)} = a_{j}^{(l)} - y_{j} = h_{\theta}(x)_{j} - y_{j}
    ```

    or in **vectorized** form:

    ``` math
    \delta^{(l)} = a^{(l)} - y = h_{\theta}(x) - y
    ```
    ***

    - For **all other layers except $l$ = 1**:
    ***
    ``` math
    \delta^{l} = (\theta^{(l)})^{T}\delta^{(l+1)} .* g'(z^{(l)})
    ```
    where
    ``` math
    g'(z^{(l))}) = a^{(l)} .* (1-a^{(l)})
    ```
    ***

2. **Gradient,  $\frac{\partial}{\partial\theta_{ij}^{(l)}} J(\theta)$**
    - The equation below **is only valid** if $\lambda = 0$ (no regularization)
    ***
    ``` math
    \frac{\partial}{\partial\theta_{ij}^{(l)}} J(\theta) = a_{j}^{(l)}\delta_{j}^{(l+1)}
    ```
    ***

#### Multiple Examples

***
1. **Set $\Delta_{ij}^{(l)} = 0$ for all (l, i, j)**
2. **For all examples, i = 1 to m:**
    1. Forward propagate to compute $a^{(i)}$ for $l = 1,2,...,L$
    2. Using $y^{(i)}$, compute $\delta^{(L)} = a^{(L)} - y^{(i)}$
    3. Compute $\delta^{(L-1)}, \delta^{L-2},...,\delta^{(2)}$
    4. Compute
        $\Delta_{ij}^{(l)} := \Delta_{ij}^{(l)} + a_{j}^{(l)}\delta_{i}^{(l+1)}$
        or **vectorized:**
        $\Delta^{(l)} := \Delta^{(l)} + \delta^{(l+1)}(a^{(l)})^{T}$
    5. Compute
    ``` math
    \frac{\partial}{\partial\theta_{ij}^{(l)}}
    J(\theta) = D_{ij}^{(l)} =   
       \begin{cases}
          \frac{1}{m} \Delta_{ij}^{(l)} + \lambda\theta_{ij}^{(l)} & if\, j \neq 0 \\
          \frac{1}{m} \Delta_{ij}^{(l)} & if\, j = 0
       \end{cases}
    ```
***

## Backpropagation in Practice

### 
