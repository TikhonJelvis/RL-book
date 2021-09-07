## Function Approximations as Vector Spaces {#sec:function-space-appendix}

### Definition of a Vector Space

A Vector space is defined as a [commutative group](https://en.wikipedia.org/wiki/Abelian_group) $\mathcal{V}$ under an addition operation (written as $+$), together with multiplication of elements of $\mathcal{V}$ with elements of a [field](https://en.wikipedia.org/wiki/Field_(mathematics)) $\mathcal{K}$ (known as scalars), expressed as a binary in-fix operation $*: \mathcal{K} \times \mathcal{V} \rightarrow \mathcal{V}$, with the following properties:

- $a * (b * \bm{v}) = (a * b) * \bm{v}$, for all $a, b \in \mathcal{K}$, for all $\bm{v} \in \mathcal{V}$.
- $1 * \bm{v} = \bm{v}$ for all $\bm{v} \in \mathcal{V}$ where $1$ denotes the multiplicative identity of $\mathcal{K}$.
- $a * (\bm{v_1} + \bm{v_2}) = a * \bm{v_1} + a * \bm{v_2}$ for all $a \in \mathcal{K}$, for all $\bm{v_1}, \bm{v_2} \in \mathcal{V}$.
- $(a + b) * \bm{v} = a * \bm{v} + b * \bm{v}$ for all $a, b \in \mathcal{K}$, for all $\bm{v} \in \mathcal{V}$.

### Definition of a Function Space

The set $\mathcal{F}$ of all functions from an arbitrary generic domain $\mathcal{X}$ to a vector space co-domain $\mathcal{V}$ (over scalars field $\mathcal{K}$) constitutes a vector space (known as function space) over the scalars field $\mathcal{K}$ with addition operation ($+$) defined as:

$$(f + g)(x) = f(x) + g(x) \text{ for all } f, g \in \mathcal{F}, \text{ for all } x \in \mathcal{X}$$

and scalar multiplication operation ($*$) defined as:

$$(a * f)(x) = a * f(x) \text{ for all } f \in \mathcal{F}, \text{ for all } a \in \mathcal{K}, \text{ for all } x \in \mathcal{X}$$

### Function Space of Linear Maps

A linear map is a function $f: \mathcal{V} \rightarrow \mathcal{W}$ where $\mathcal{V}$ is a vector space over a scalars field $\mathcal{K}$ and $\mathcal{W}$ is a vector space over the same scalars field $\mathcal{K}$, having the following two properties:

- $f(\bm{v_1} + \bm{v_2}) = f(\bm{v_1}) + f(\bm{v_2})$ for all $\bm{v_1}, \bm{v_2} \in \mathcal{V}$ (i.e., application of $f$ commutes with the addition operation).
- $f(a * \bm{v}) = a * f(\bm{v})$ for all $\bm{v} \in \mathcal{V}$, for all $a \in \mathcal{K}$ (i.e., applications of $f$ commutes with the scalar multiplication operation).

Then the set of all linear maps with domain $\mathcal{V}$ and co-domain $\mathcal{W}$ constitute a function space (restricted to just this subspace of all linear maps, rather than the space of all $\mathcal{V} \rightarrow \mathcal{W}$ functions). This function space (restricted to the subspace of all $\mathcal{V} \rightarrow \mathcal{W}$ linear maps) is denoted as the vector space $\mathcal{L}(\mathcal{V}, \mathcal{W})$.

The specialization of the function space of linear maps to the space $\mathcal{L}(\mathcal{V}, \mathcal{K})$ (i.e., specializing the vector space $\mathcal{W}$ to the scalars field $\mathcal{K}$) is known as the dual vector space and is denoted as $\mathcal{V}^*$.

### Vector Spaces in Function Approximations

We represent function approximations by parameterized functions $f: \mathcal{X} \times D[\mathbb{R}] \rightarrow \mathbb{R}$ where $\mathcal{X}$ is the input domain and $D[\mathbb{R}]$ is the parameters domain. The notation $D[Y]$ refers to a generic container data type $D$ over a component generic data type $Y$. The data type $D$ is specified as a generic container type because we consider generic function approximations here. A specific family of function approximations will customize to a specific container data type for $D$ (eg: linear function approximations will customize $D$ to a Sequence data type, a feed-forward deep neural network will customize $D$ to a Sequence of 2-dimensional arrays). We consider 2 different Vector Spaces relevant to Function Approximations:

#### Parameters Space $\mathcal{P}$

$D[\mathbb{R}]$ forms a vector space $\mathcal{P}$ over the scalars field $\mathbb{R}$ with addition operation defined as element-wise real-numbered addition and scalar multiplication operation defined as element-wise multiplication with real-numbered scalars. We refer to this vector space $\mathcal{P}$ as the *Parameters Space*.

#### Representational Space $\mathcal{G}$

We consider a function $I: \mathcal{P} \rightarrow (\mathcal{X} \rightarrow \mathbb{R})$ defined as $I(\bm{w}) = g: \mathcal{X} \rightarrow \mathbb{R}$ for all $\bm{w} \in \mathcal{P}$ such that $g(x) = f(x, \bm{w})$ for all $x \in \mathcal{X}$. The *Range* of this function $I$ forms a vector space $\mathcal{G}$ over the scalars field $\mathbb{R}$ with addition operation defined as:

$$I(\bm{w_1}) + I(\bm{w_2}) = I(\bm{w_1} + \bm{w_2}) \text{ for all } \bm{w_1}, \bm{w_2} \in \mathcal{P}$$

and multiplication operation defined as:

$$a * I(\bm{w}) = I(a * \bm{w}) \text{ for all } \bm{w} \in \mathcal{P}, \text{ for all } a \in \mathbb{R}$$

We refer to this vector space $\mathcal{G}$ as the *Representational Space* (to signify the fact that addition and multiplication operations in $\mathcal{G}$ essentially "delegate" to addition and multiplication operations in the Parameters Space $\mathcal{P}$, with any parameters $\bm{w} \in \mathcal{P}$ serving as the internal representation of a function approximation $I(\bm{w}): \mathcal{X} \rightarrow \mathbb{R}$). This "delegation" from $\mathcal{G}$ to $\mathcal{P}$ implies that $I$ is a linear map from Parameters Space $\mathcal{P}$ to Representational Space $\mathcal{G}$.

### The Gradient Function

The gradient of a function approximation $f: \mathcal{X} \times D[\mathbb{R}] \rightarrow \mathbb{R}$ with respect to parameters $\bm{w} \in D[\mathbb{R}]$ (denoted as $\nabla_{\bm{w}} f(x, \bm{w})$) is an element of the parameters domain $D[\mathbb{R}]$. By treating both $\bm{w}$ and $\nabla_{\bm{w}} f(x, \bm{w})$ as vectors in the Parameters Space $\mathcal{P}$, we define the gradient function
$$G: \mathcal{X} \rightarrow (\mathcal{P} \rightarrow \mathcal{P})$$
as:
$$G(x)(\bm{w}) = \nabla_{\bm{w}} f(x, \bm{w})$$
for all $x \in \mathcal{X}$, for all $\bm{w} \in \mathcal{P}$.

### Linear Function Approximations

If we restrict to linear function approximations, for all $x \in \mathcal{X}$,
$$f(x, \bm{w}) = h(\bm{w}) = \bm{\Phi}(x)^T \cdot \bm{w}$$
where $\bm{w} \in \mathbb{R}^m = \mathcal{P}$ and $\bm{\Phi}: \mathcal{X} \rightarrow \mathbb{R}^m$ represents the feature functions (note: $\bm{\Phi}(x)^T \cdot \bm{w}$ is the usual inner-product in the vector space $\mathbb{R}^m$).

Then the gradient function $G: \mathcal{X} \rightarrow (\mathbb{R}^m \rightarrow \mathbb{R}^m)$ can be written as:
$$G(x)(\bm{w}) = \nabla_{\bm{w}} (\bm{\Phi}(x)^T \cdot \bm{w}) = \bm{\Phi}(x)$$
for all $x \in \mathcal{X}$, for all $\bm{w} \in \mathbb{R}^m$.

Also note that in the case of linear function approximations, the function $I: \mathcal{R}^m \rightarrow (\mathcal{X} \rightarrow \mathbb{R})$ is a linear map from $\mathcal{R}^m = \mathcal{P}$ to a vector subspace of the function space $\mathcal{F}$ of all $\mathcal{X} \rightarrow \mathbb{R}$ functions over scalars field $\mathbb{R}$ (with pointwise operations). This is because for all $x \in \mathcal{X}$:

$$\bm{\Phi}(x)^T \cdot (\bm{w}_1 + \bm{w}_2) = \bm{\Phi}(x)^T \cdot \bm{w}_1 + \bm{\Phi}(x)^T \cdot \bm{w}_2 \text{ for all } \bm{w}_1, \bm{w}_2 \in \mathbb{R}^m$$
$$\bm{\Phi}(x)^T \cdot (a * \bm{w}) = a * (\bm{\Phi}(x)^T \cdot \bm{w}) \text{ for all } \bm{w} \in \mathbb{R}^m, \text{ for all } a \in \mathbb{R}$$

The key concept here is that for the case of linear function approximations, addition and multiplication "delegating" operations in $\mathcal{G}$ coincide with addition and multiplication "pointwise" operations in $\mathcal{F}$, which implies that $\mathcal{G}$ is isomorphic to a vector subspace of $\mathcal{F}$.

### Stochastic Gradient Descent

Stochastic Gradient Descent is a function

$$SGD: \mathcal{X} \times \mathbb{R} \rightarrow (\mathcal{P} \rightarrow \mathcal{P})$$

representing a mapping from (predictor, response) data to a "parameters-update" function (in order to improve the function approximation), defined as:

$$SGD(x, y)(\bm{w}) = \bm{w} + (- \alpha \cdot (f(x, \bm{w}) - y) * G(x)(\bm{w}))$$
for all $x \in \mathcal{X}, y \in \mathbb{R}, \bm{w} \in \mathcal{P}$, where $\alpha \in \mathbb{R}^+$ represents the learning rate (step size of SGD).

For a fixed data pair $(x, y) \in \mathcal{X} \times \mathbb{R}$, with prediction error function $e: \mathcal{P} \rightarrow \mathbb{R}$ defined as $e(\bm{w}) = y - f(x, \bm{w})$, the (SGD-based) parameters change function (function from parameters to change in parameters)

$$U: \mathcal{P} \rightarrow \mathcal{P}$$

is defined as:

$$U(\bm{w}) = SGD(x, y)(\bm{w}) + (-1 * \bm{w}) = \alpha * (e(\bm{w}) * G(x)(\bm{w}))$$

for all $\bm{w} \in \mathcal{P}$.

So, we can conceptualize the parameters change function $U$ as the product of:

- Learning rate $\alpha \in \mathbb{R}^+$
- Prediction error function $e: \mathcal{P} \rightarrow \mathbb{R}$
- Gradient operator $G(x): \mathcal{P} \rightarrow \mathcal{P}$

Note that the product of functions $e$ and $G(x)$ above is element-wise in their common domain $\mathcal{P} = D[\mathbb{R}]$, resulting in the scalar ($\mathbb{R}$) multiplication of vectors in $\mathcal{P}$.

Updating vector $\bm{w}$ to vector $\bm{w} + U(\bm{w})$ in the Parameter Space $\mathcal{P}$ results in updating function $I(\bm{w}): \mathcal{X} \rightarrow \mathbb{R}$ to function $I(\bm{w} + U(\bm{w})): \mathcal{X} \rightarrow \mathbb{R}$ in the Representational Space $\mathcal{G}$. This is rather convenient since we can view all of the above addition/multiplication operations in the Parameter Space $\mathcal{P}$ as addition/multiplication operations in the Representational Space $\mathcal{G}$.

### SGD Update for Linear Function Approximations

As a reminder, for the case of linear function approximations, $\mathcal{G}$ is isomorphic to a vector subspace of the function space $\mathcal{F}$ of all $\mathcal{X} \rightarrow \mathbb{R}$ functions over the scalars field $\mathbb{R}$ (addition and multiplication "delegating" operations in $\mathcal{G}$ coincide with addition and multiplication "pointwise" operations in $\mathcal{F}$).

So in the case of linear function approximations, when updating vector $\bm{w}$ to vector $\bm{w} + \alpha * ((y - \bm{\Phi}(x)^T \cdot \bm{w}) * \bm{\Phi}(x))$ in the Parameter Space $\mathcal{P} = \mathbb{R}^m$, applying the linear map $I: \mathbb{R}^m \rightarrow \mathcal{G}$ updates functions in $\mathcal{G}$ with corresponding pointwise addition and multiplication operations.

Concretely, a linear function approximation $g: \mathcal{X} \rightarrow \mathbb{R}$ defined as $g(z) = \bm{\Phi}(z)^T \cdot \bm{w}$ for all $z \in \mathcal{X}$ updates correspondingly to the function $g^{(x,y)}: \mathcal{X} \rightarrow \mathbb{R}$ defined as $g^{(x,y)}(z) = \bm{\Phi}(z)^T \cdot \bm{w} + \alpha \cdot (y - \bm{\Phi}(x)^T \cdot \bm{w}) \cdot (\bm{\Phi}(z)^T \cdot \bm{\Phi}(x))$ for all $z \in \mathcal{X}$.

It's useful to note that the change in the evaluation at $z \in \mathcal{X}$ is simply the product of:

- Learning rate $\alpha \in \mathbb{R}^+$
- Prediction Error $y - \bm{\Phi}(x)^T \cdot \bm{w} \in \mathbb{R}$ for the updating data $(x,y) \in \mathcal{X} \times \mathbb{R}$
- Inner-product of the feature vector $\bm{\Phi}(x) \in \mathbb{R}^m$ of the updating input value $x \in \mathcal{X}$ and the feature vector $\bm{\Phi}(z) \in \mathbb{R}^m$ of the evaluation input value $z \in \mathcal{X}$.
