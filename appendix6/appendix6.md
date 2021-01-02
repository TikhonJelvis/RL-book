## Function Approximations as Vector Spaces {#sec:function-space-appendix}

### Definition of a Vector Space

A Vector space is defined as a commutative group $\mathcal{V}$ under an addition operation (written as $+$), together with multiplication of elements of $\mathcal{V}$ with elements of a field $\mathcal{K}$ (known as scalars), expressed as a binary in-fix operation $\cdot: \mathcal{K} \times \mathcal{V} \rightarrow \mathcal{V}$, with the following properties:

- $a \cdot (b \cdot \bm{v}) = (a \cdot b) \cdot \bm{v}$, for all $a, b \in \mathcal{K}$, for all $\bm{v} \in \mathcal{V}$.
- $1 \cdot \bm{v} = \bm{v}$ for all $\bm{v} \in \mathcal{V}$ where $1$ denotes the multiplicative identity of $\mathcal{K}$.
- $a \cdot (\bm{v_1} + \bm{v_2}) = a \cdot \bm{v_1} + a \cdot \bm{v_2}$ for all $a \in \mathcal{K}$, for all $\bm{v_1}, \bm{v_2} \in \mathcal{V}$.
- $(a + b) \cdot \bm{v} = a \cdot \bm{v} + b \cdot \bm{v}$ for all $a, b \in \mathcal{K}$, for all $\bm{v} \in \mathcal{V}$.

### Definition of a Function Space

The set $\mathcal{F}$ of all functions from an arbitrary generic domain $\mathcal{X}$ to a vector space co-domain $\mathcal{V}$ (over scalars field $\mathcal{K}$) constitutes a vector space (known as function space) over the scalars field $\mathcal{K}$ with addition operation ($+$) defined as:

$$(f + g)(x) = f(x) + g(x) \text{ for all } f, g \in \mathcal{F}, \text{ for all } x \in \mathcal{X}$$

and scalar multiplication operation ($\cdot$) defined as:

$$(a \cdot f)(x) = a \cdot f(x) \text{ for all } f \in \mathcal{F}, \text{ for all } a \in \mathcal{K}, \text{ for all }x \in \mathcal{X}$$

### Function Space of Linear Maps

A linear map is a function $f: \mathcal{V} \rightarrow \mathcal{W}$ where $\mathcal{V}$ is a vector space over a scalars field $\mathcal{K}$ and $\mathcal{W}$ is a vector space over the same scalars field $\mathcal{K}$, having the following two properties:

- $f(\bm{v_1} + \bm{v_2}) = f(\bm{v_1}) + f(\bm{v_2})$ for all $\bm{v_1}, \bm{v_2} \in \mathcal{V}$ (i.e., application of $f$ commutes with the addition operation).
- $f(a \cdot \bm{v}) = a \cdot f(\bm{v})$ for all $\bm{v} \in \mathcal{V}$, for all $a \in \mathcal{K}$ (i.e., applications of $f$ commutes with the scalar multiplication operation).

Then the set of all linear maps with domain $\mathcal{V}$ and co-domain $\mathcal{W}$ constitute a function space (restricted to just this subspace of all linear maps, rather than the space of all $\mathcal{V} \rightarrow \mathcal{W}$ functions). This function space (restricted to the subspace of all $\mathcal{V} \rightarrow \mathcal{W}$ linear maps) is denoted as the vector space $\mathcal{L}(\mathcal{V}, \mathcal{W})$.

The specialization of the function space of linear maps to the space $\mathcal{L}(\mathcal{V}, \mathcal{K})$ (i.e., specializing the vector space $\mathcal{W}$ to the scalars field $\mathcal{K}$) is known as the dual vector space and is denoted as $\mathcal{V}^*$.

### Vector Spaces in Function Approximations

We represent function approximations by parameterized functions $f: \mathcal{X} \times D[\mathbb{R}] \rightarrow \mathbb{R}$ where $\mathcal{X}$ is the input domain and $D[\mathbb{R}]$ is the parameters domain. The notation $D[Y]$ refers to a generic container data type $D$ over a component generic data type $Y$. The data type $D$ is specified as a generic container type because we consider generic function approximations here. A specific family of function approximations will customize to a specific container data type for $D$ (eg: linear function approximations will customize $D$ to a Sequence data type, a feed-forward deep neural network will customize $D$ to a Sequence of 2-dimensional arrays). We consider 3 different Vector Spaces:

#### Vector Space $\mathcal{V}_1$

$D[\mathbb{R}]$ is a vector space $\mathcal{V}_1$ over the scalars field $\mathbb{R}$ with addition operation defined as element-wise real-numbered addition and scalar multiplication operation defined as element-wise multiplication with real-numbered scalars.

#### Vector Space $\mathcal{V}_2$

For fixed parameters $\bm{w} \in D[\mathbb{R}] = \mathcal{V}_1$, the set of functions $g: \mathcal{X} \rightarrow \mathbb{R}$ defined as $g(x) = f(x, \bm{w})$ for all $x \in \mathcal{X}$ form a function space $\mathcal{V}_2$ over the scalars field $\mathbb{R}$. We note that a vector in $\mathcal{V}_1$ naturally specifies a vector in $\mathcal{V}_2$ with the mapping $I: \mathcal{V}_1 \rightarrow \mathcal{V}_2$ defined as $I(\bm{w}) = g: \mathcal{X} \rightarrow \mathbb{R}$ for all $\bm{w} \in \mathcal{V}_1$ such that $g(x) = f(x, \bm{w})$ for all $x \in \mathcal{X}$.

#### Vector Space $\mathcal{V}_3$

For fixed input $x \in \mathcal{X}$, the set of functions $h: \mathcal{V}_1 \rightarrow \mathbb{R}$ defined as $h(\bm{w}) = f(x, \bm{w})$ for all $\bm{w} \in D[\mathbb{R}] = \mathcal{V}_1$ form a function space $\mathcal{V}_3$ over the scalars field $\mathbb{R}$. Notationally, the vector space $\mathcal{V}_3$ refers to the function space of $\mathcal{V}_1 \rightarrow \mathbb{R}$ functions. We note that an input $x \in \mathcal{X}$ naturally specifies a vector in $\mathcal{V}_3$ with the mapping $J: \mathcal{X} \rightarrow \mathcal{V}_3$ defined as: $J(x) = h: \mathcal{V}_1 \rightarrow \mathbb{R}$ for all $x \in \mathcal{X}$ such that $h(\bm{w}) = f(x, \bm{w})$ for all $\bm{w} \in \mathcal{V}_1$.

### The Gradient Function

The gradient of $f$ with respect to parameters $\bm{w} \in D[\mathbb{R}]$ (denoted as $\nabla_{\bm{w}} f(x, \bm{w})$) is an element of the parameters domain $D[\mathbb{R}]$, and hence can be treated as a vector in the Vector Space $\mathcal{V}_1$. So we define the gradient function
$$G: \mathcal{X} \rightarrow (\mathcal{V}_1 \rightarrow \mathcal{V}_1)$$
as:
$$G(x)(\bm{w}) = \nabla_{\bm{w}} f(x, \bm{w})$$
for all $x \in \mathcal{X}$, for all $\bm{w} \in \mathcal{V}_1$.

### Linear Function Approximations

If we restrict to linear function approximations
$$f(x, \bm{w}) = h(\bm{w}) = \bm{\Phi}(x) \circ \bm{w}$$
where $\bm{w} \in \mathbb{R}^m$ and $\bm{\Phi}: \mathcal{X} \rightarrow \mathbb{R}^m$ represents the feature functions (with $\circ$ denoting inner-product in the vector space $\mathbb{R}^m$), then we can restrict to the function space $\mathcal{L}(\mathbb{R}^m, \mathbb{R})$ of all linear maps $h: \mathbb{R}^m \rightarrow \mathbb{R}$ (i.e., the dual vector space ${(\mathbb{R}^m)}^*$ of the vector space $\mathbb{R}^m$). This is a subspace of the function space $\mathcal{V}_3$ (i.e., a subspace of the function space of all $\mathbb{R}^m \rightarrow \mathbb{R}$ functions). Then, the gradient function $G: \mathcal{X} \rightarrow (\mathbb{R}^m \rightarrow \mathbb{R}^m)$ can be written as:
$$G(x)(\bm{w}) = \nabla_{\bm{w}} (\bm{\Phi}(x) \circ \bm{w}) = \bm{\Phi}(x)$$
for all $x \in \mathcal{X}$, for all $\bm{w} \in \mathbb{R}^m$.

Also note that in the case of linear function approximations, the mapping $I: \mathcal{V}_1 \rightarrow \mathcal{V}_2$ defined above is a linear map $I: \mathbb{R}^m \rightarrow (\mathcal{X} \rightarrow \mathbb{R})$ since $\bm{\Phi}(x) \circ (\bm{w}_1 + \bm{w}_2) = \bm{\Phi}(x) \circ \bm{w}_1 + \bm{\Phi}(x) \circ \bm{w}_2$ for all $\bm{w}_1, \bm{w}_2 \in \mathbb{R}^m$ and $\bm{\Phi}(x) \circ (a \cdot \bm{w}) = a \cdot (\bm{\Phi}(x) \circ \bm{w})$ for all $\bm{w} \in \mathbb{R}^m$, for all $a \in \mathbb{R}$.

### Stochastic Gradient Descent

Stochastic Gradient Descent is a function

$$SGD: \mathcal{X} \times \mathbb{R} \rightarrow (\mathcal{V}_1 \rightarrow \mathcal{V}_1)$$

representing a mapping from (predictor, response) data to a "parameters-update" function (in order to improve the function approximation), defined as:

$$SGD(x, y)(\bm{w}) = \bm{w} - \alpha \cdot (f(x, \bm{w}) - y) \cdot G(x)(\bm{w})$$
for all $x \in \mathcal{X}, y \in \mathbb{R}, \bm{w} \in \mathcal{V}_1$, where $\alpha \in \mathbb{R}^+$ represents the learning rate (step size of SGD).

For a fixed data pair $(x, y) \in \mathcal{X} \times \mathbb{R}$, with prediction error function $e: \mathcal{V}_1 \rightarrow \mathbb{R}$ defined as $e(\bm{w}) = y - f(x, \bm{w})$, the (SGD-based) parameters change function (function from parameters to change in parameters)

$$U: \mathcal{V}_1 \rightarrow \mathcal{V}_1$$

is defined as:

$$U(\bm{w}) = SGD(x, y)(\bm{w}) - \bm{w} = \alpha \cdot e(\bm{w}) \cdot G(x)(\bm{w})$$

for all $\bm{w} \in \mathcal{V}_1$.

So, we can conceptualize the parameters change function $U$ as the product of:

- Learning rate $\alpha \in \mathbb{R}^+$
- Prediction error function $e: \mathcal{V}_1 \rightarrow \mathbb{R}$
- Gradient operator $G(x): \mathcal{V}_1 \rightarrow \mathcal{V}_1$

Note that the product of functions $e$ and $G(x)$ above is element-wise in their common domain $\mathcal{V}_1 = D[\mathbb{R}]$, resulting in the scalar ($\mathbb{R}$) multiplication of vectors in $\mathcal{V}_1$.

Updating vector $\bm{w}$ to vector $\bm{w} + U(\bm{w})$ in the vector space $\mathcal{V}_1$ results in updating function $I(\bm{w}): \mathcal{X} \rightarrow \mathbb{R}$ to function $I(\bm{w} + U(\bm{w})): \mathcal{X} \rightarrow \mathbb{R}$ in the function space $\mathcal{V}_2$.

### SGD Update for Linear Function Approximation

As a reminder, $I$ is a linear map in the case of linear function approximation. In the case of linear function approximation, when updating vector $\bm{w}$ to vector $\bm{w} + \alpha \cdot (y - \bm{\Phi}(x) \circ \bm{w}) \cdot \bm{\Phi}(x)$ in the vector space $\mathcal{V}_1 = \mathbb{R}^m$, the linear map $I: \mathbb{R}^m \rightarrow \mathcal{V}_2$ updates function $g: \mathcal{X} \rightarrow \mathbb{R}$ defined as $g(z) = \bm{\Phi}(z) \circ \bm{w}$ for all $z \in \mathcal{X}$ to the function $g^{(x,y)}: \mathcal{X} \rightarrow \mathbb{R}$ defined as $g^{(x,y)}(z) = \bm{\Phi}(z) \circ \bm{w} + \alpha \cdot (y - \bm{\Phi}(x) \circ \bm{w}) \cdot (\bm{\Phi}(z) \circ \bm{\Phi}(x))$ for all $z \in \mathcal{X}$. It's useful to note that the change in the evaluation at $z \in \mathcal{X}$ is the product of:

- Learning rate $\alpha \in \mathbb{R}^+$
- Prediction Error $y - \bm{\Phi}(x) \circ \bm{w} \in \mathbb{R}$ for the updating data $(x,y) \in \mathcal{X} \times \mathbb{R}$
- Inner-product of the feature vector $\bm{\Phi}(x) \in \mathbb{R}^m$ of the updating input value $x \in \mathcal{X}$ and the feature vector $\bm{\Phi}(z) \in \mathbb{R}^m$ of the evaluation input value $z \in \mathcal{X}$.
