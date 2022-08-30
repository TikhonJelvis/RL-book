## Function Approximations as Affine Spaces {#sec:function-space-appendix}

### Vector Space

\index{vector space|textbf}
\index{commutative group|textbf}

A Vector space is defined as a [commutative group](https://en.wikipedia.org/wiki/Abelian_group) $\mathcal{V}$ under an addition operation (written as $+$), together with multiplication of elements of $\mathcal{V}$ with elements of a [field](https://en.wikipedia.org/wiki/Field_(mathematics)) $\mathcal{K}$ (known as scalars field), expressed as a binary in-fix operation $*: \mathcal{K} \times \mathcal{V} \rightarrow \mathcal{V}$, with the following properties:

* $a * (b * \bm{v}) = (a * b) * \bm{v}$, for all $a, b \in \mathcal{K}$, for all $\bm{v} \in \mathcal{V}$.
* $1 * \bm{v} = \bm{v}$ for all $\bm{v} \in \mathcal{V}$ where $1$ denotes the multiplicative identity of $\mathcal{K}$.
* $a * (\bm{v_1} + \bm{v_2}) = a * \bm{v_1} + a * \bm{v_2}$ for all $a \in \mathcal{K}$, for all $\bm{v_1}, \bm{v_2} \in \mathcal{V}$.
* $(a + b) * \bm{v} = a * \bm{v} + b * \bm{v}$ for all $a, b \in \mathcal{K}$, for all $\bm{v} \in \mathcal{V}$.

### Function Space

\index{functions!function space|textbf}

The set $\mathcal{F}$ of all functions from an arbitrary generic domain $\mathcal{X}$ to a vector space co-domain $\mathcal{V}$ (over scalars field $\mathcal{K}$) constitutes a vector space (known as function space) over the scalars field $\mathcal{K}$ with addition operation ($+$) defined as:

$$(f + g)(x) = f(x) + g(x) \text{ for all } f, g \in \mathcal{F}, \text{ for all } x \in \mathcal{X}$$
and scalar multiplication operation ($*$) defined as:
$$(a * f)(x) = a * f(x) \text{ for all } f \in \mathcal{F}, \text{ for all } a \in \mathcal{K}, \text{ for all } x \in \mathcal{X}$$

Hence, addition and scalar multiplication for a function space are defined point-wise.

### Linear Map of Vector Spaces

\index{vector space!linear map|textbf}

A linear map of Vector Spaces is a function $h: \mathcal{V} \rightarrow \mathcal{W}$ where $\mathcal{V}$ is a vector space over a scalars field $\mathcal{K}$ and $\mathcal{W}$ is a vector space over the same scalars field $\mathcal{K}$, having the following two properties:

* $h(\bm{v_1} + \bm{v_2}) = h(\bm{v_1}) + h(\bm{v_2})$ for all $\bm{v_1}, \bm{v_2} \in \mathcal{V}$ (i.e., application of $h$ commutes with the addition operation).
* $h(a * \bm{v}) = a * h(\bm{v})$ for all $\bm{v} \in \mathcal{V}$, for all $a \in \mathcal{K}$ (i.e., application of $h$ commutes with the scalar multiplication operation).

Then the set of all linear maps with domain $\mathcal{V}$ and co-domain $\mathcal{W}$ constitute a function space (restricted to just this subspace of all linear maps, rather than the space of all $\mathcal{V} \rightarrow \mathcal{W}$ functions) that we denote as $\mathcal{L}(\mathcal{V}, \mathcal{W})$.

The specialization of the function space of linear maps to the space $\mathcal{L}(\mathcal{V}, \mathcal{K})$ (i.e., specializing the vector space $\mathcal{W}$ to the scalars field $\mathcal{K}$) is known as the dual vector space and is denoted as $\mathcal{V}^*$.

### Affine Space

\index{affine space|textbf}

An Affine Space is defined as a set $\mathcal{A}$ associated with a vector space $\mathcal{V}$ and a binary in-fix operation $\oplus: \mathcal{A} \times \mathcal{V} \rightarrow \mathcal{A}$, with the following properties:

* For all $\bm{a} \in \mathcal{A}, \bm{a} \oplus 0 = \bm{a}$, where $0$ is the zero vector in $\mathcal{V}$ (this is known as the right identity property).
* For all $\bm{v_1}, \bm{v_2} \in \mathcal{V}$, for all $\bm{a} \in \mathcal{A}, (\bm{a} \oplus \bm{v_1}) \oplus \bm{v_2} = \bm{a} \oplus (\bm{v_1} + \bm{v_2})$ (this is known as the associativity property).
* For each $\bm{a} \in \mathcal{A}$, the mapping $f_{\bm{a}} : \mathcal{V} \rightarrow \mathcal{A}$ defined as $f_{\bm{a}}(\bm{v}) = \bm{a} \oplus \bm{v}$ for all $\bm{v} \in \mathcal{V}$ is a bijection (i.e., one-to-one and onto mapping).

The elements of an affine space are called *points* and the elements of the vector space associated with an affine space are called *translations*. The idea behind affine spaces is that unlike a vector space, an affine space doesn't have a notion of a zero element and one cannot add two *point*s in the affine space. Instead one adds a *translation* (from the associated vector space) to a *point* (from the affine space) to yield another *point* (in the affine space). The term *translation* is used to signify that we "translate" (i.e. shift) a point to another point in the affine space with the shift being effected by a *translation* in the associated vector space. The bijection property defined above implies that there is a notion of "subtracting" one *point* of the affine space from another *point* of the affine space (denoted with the operation $\ominus$), yielding a *translation* in the associated vector space. Formally, $\ominus$ is defined as:
$$\text{For each } \bm{a_1}, \bm{a_2} \in \mathcal{A}, \text{ there exists a unique } \bm{v} \in \mathcal{V}, \text{ denoted } \bm{a_2} \ominus \bm{a_1}, \text{ such that } \bm{a_2} = \bm{a_1} \oplus \bm{v}$$
A simple way to visualize an affine space is by considering the simple example of the affine space of all 3-D points on the plane defined by the equation $z=1$, i.e., the set of all points $(x,y,1)$ for all $x \in \mathbb{R}, y \in \mathbb{R}$. The associated vector space is the set of all 3-D points on the plane defined by the equation $z=0$, i.e., the set of all points $(x,y,0)$ for all $x \in \mathbb{R}, y \in \mathbb{R}$ (with element-wise addition and scalar multiplication operations). The $\oplus$ operation is element-wise addition. We see that any point $(x,y,1)$ on the affine space is *translated* to the point $(x+x',y+y',1)$ by the translation $(x',y',0)$ in the vector space. Note that the translation $(0,0,0)$ (zero vector) results in the point $(x,y,1)$ remaining unchanged. Note that translations $(x',y',0)$ and $(x'',y'',0)$ applied one after the other is the same as the single translation $(x'+x'',y'+y'',0)$. Finally, note that for any fixed point $(x,y,1)$, we have a bijective mapping from the vector space $z=0$ to the affine space $z=1$ that maps any translation $(x',y',0)$ to the point $(x+x',y+y',1)$.

### Affine Map

\index{affine map|textbf}

An Affine Map is a function $h: \mathcal{A} \rightarrow \mathcal{B}$, associated with a linear map $l: \mathcal{V} \rightarrow \mathcal{W}$, where $\mathcal{A}$ is an affine space associated with vector space $\mathcal{V}$ and $\mathcal{B}$ is an affine space associated with vector space $\mathcal{W}$, having the following property:
$$h(\bm{a_1}) \ominus h(\bm{a_2}) = l(\bm{a_1} \ominus \bm{a_2}) \text{ for all } \bm{a_1}, \bm{a_2} \in \mathcal{A}$$
This implies:
$$h(\bm{a} \oplus \bm{v}) = h(\bm{a}) \oplus l(\bm{v}) \text{ for all } \bm{a} \in \mathcal{A}, \text{ for all } \bm{v} \in \mathcal{V}$$
The intuitive way of thinking about an affine map $h$ is that it's completely defined by the image $h(\bm{a})$ of *any single point* $\bm{a} \in \mathcal{A}$ and by it's associated linear map $l$.

Later in this appendix, we consider a specialization of affine maps—when $\mathcal{V} = \mathcal{W}$ and $l$ is the identity function. For this specialization, we have:
$$h(\bm{a} \oplus \bm{v}) = h(\bm{a}) \oplus \bm{v} \text{ for all } \bm{a} \in \mathcal{A}, \text{ for all } \bm{v} \in \mathcal{V}$$
The way to think about this is that $\oplus: \mathcal{A} \times \mathcal{V} \rightarrow \mathcal{A}$ simply delegates to $\oplus: \mathcal{B} \times \mathcal{V} \rightarrow \mathcal{B}$. So we shall refer to such a specialization of affine maps as *Delegating Map* and the corresponding affine spaces $\mathcal{A}$ and $\mathcal{B}$ as *Delegator Space* and *Delegate Space*, respectively.


### Function Approximations

\index{function approximation|(}

We represent function approximations by parameterized functions $f: \mathcal{X} \times D[\mathbb{R}] \rightarrow \mathbb{R}$ where $\mathcal{X}$ is the input domain and $D[\mathbb{R}]$ is the parameters domain. The notation $D[Y]$ refers to a generic container data type $D$ over a component generic data type $Y$. The data type $D$ is specified as a generic container data type because we consider generic function approximations here. A specific family of function approximations will customize to a specific container data type for $D$ (e.g., linear function approximations will customize $D$ to a Sequence data type, a feed-forward deep neural network will customize $D$ to a Sequence of 2-dimensional arrays). We are interested in viewing Function Approximations as *point*s in an appropriate Affine Space. To explain this, we start by viewing parameters as *point*s in an Affine Space.

#### $D[\mathbb{R}]$ as an Affine Space $\mathcal{P}$

\index{affine space}
\index{function approximation!gradient descent}

When performing Stochastic Gradient Descent or Batch Gradient Descent, parameters $\bm{p} \in D[\mathbb{R}]$ of a function approximation $f: \mathcal{X} \times D[\mathbb{R}] \rightarrow \mathbb{R}$ are updated using an appropriate linear combination of gradients of $f$ with respect to $\bm{p}$ (at specific values of $x \in \mathcal{X}$). Hence, the parameters domain $D[\mathbb{R}]$ can be treated as an affine space (call it $\mathcal{P}$) whose associated vector space (over scalars field $\mathbb{R}$) is the set of gradients of $f$ with respect to parameters $\bm{p} \in D[\mathbb{R}]$ (denoted as $\nabla_{\bm{p}} f(x, \bm{p})$), evaluated at specific values of $x \in \mathcal{X}$, with addition operation defined as element-wise real-numbered addition and scalar multiplication operation defined as element-wise multiplication with real-numbered scalars. We refer to this Affine Space $\mathcal{P}$ as the *Parameters Space* and we refer to its associated vector space (of gradients) as the *Gradient Space* $\mathcal{G}$. Since each *point* in $\mathcal{P}$ and each *translation* in $\mathcal{G}$ is an element in $D[\mathbb{R}]$, the $\oplus$ operation is element-wise real-numbered addition.

We define the gradient function
$$G: \mathcal{X} \rightarrow (\mathcal{P} \rightarrow \mathcal{G})$$
as:
$$G(x)(\bm{p}) = \nabla_{\bm{p}} f(x, \bm{p})$$
for all $x \in \mathcal{X}$, for all $\bm{p} \in \mathcal{P}$.

#### Delegator Space $\mathcal{R}$

We consider a function $I: \mathcal{P} \rightarrow (\mathcal{X} \rightarrow \mathbb{R})$ defined as $I(\bm{p}) = g: \mathcal{X} \rightarrow \mathbb{R}$ for all $\bm{p} \in \mathcal{P}$ such that $g(x) = f(x, \bm{p})$ for all $x \in \mathcal{X}$. The *Range* of this function $I$ forms an affine space $\mathcal{R}$ whose associated vector space is the Gradient Space $\mathcal{G}$, with the $\oplus$ operation defined as: 

$$I(\bm{p}) \oplus \bm{v} = I(\bm{p} \oplus \bm{v}) \text{ for all } \bm{p} \in \mathcal{P}, \bm{v} \in \mathcal{G}$$

We refer to this affine space $\mathcal{R}$ as the *Delegator Space* to signify the fact that the $\oplus$ operation for $\mathcal{R}$ simply "delegates" to the $\oplus$ operation for $\mathcal{P}$ and so, the parameters $\bm{p} \in \mathcal{P}$ basically serve as the internal representation of the function approximation $I(\bm{p}): \mathcal{X} \rightarrow \mathbb{R}$. This "delegation" from $\mathcal{R}$ to $\mathcal{P}$ implies that $I$ is a *Delegating Map* (as defined earlier) from Parameters Space $\mathcal{P}$ to Delegator Space $\mathcal{R}$.

Notice that the `__add__` method of the `Gradient` class in [rl/function_approx.py](https://github.com/TikhonJelvis/RL-book/blob/master/rl/function_approx.py) is overloaded. One of the `__add__` methods corresponds to vector addition of two gradients in the Gradient Space $\mathcal{G}$. The other `__add__` method corresponds to the $\oplus$ operation adding a gradient (treated as a *translation* in the vector space of gradients) to a function approximation (treated as a *point* in the affine space of function approximations).

### Stochastic Gradient Descent

\index{function approximation!gradient descent}

Stochastic Gradient Descent is a function

$$SGD: \mathcal{X} \times \mathbb{R} \rightarrow (\mathcal{P} \rightarrow \mathcal{P})$$
representing a mapping from (predictor, response) data to a "parameters-update" function (in order to improve the function approximation), defined as:

$$SGD(x, y)(\bm{p}) = \bm{p} \oplus (\alpha * ((y - f(x, \bm{p})) * G(x)(\bm{p})))$$
for all $x \in \mathcal{X}, y \in \mathbb{R}, \bm{p} \in \mathcal{P}$, where $\alpha \in \mathbb{R}^+$ represents the learning rate (step size of SGD).

For a fixed data pair $(x, y) \in \mathcal{X} \times \mathbb{R}$, with prediction error function $e: \mathcal{P} \rightarrow \mathbb{R}$ defined as $e(\bm{p}) = y - f(x, \bm{p})$, the (SGD-based) parameters change function

$$U: \mathcal{P} \rightarrow \mathcal{G}$$

is defined as:

$$U(\bm{p}) = SGD(x, y)(\bm{p}) \ominus \bm{p} = \alpha * (e(\bm{p}) * G(x)(\bm{p}))$$

for all $\bm{p} \in \mathcal{P}$.

So, we can conceptualize the parameters change function $U$ as the product of:

* Learning rate $\alpha \in \mathbb{R}^+$
* Prediction error function $e: \mathcal{P} \rightarrow \mathbb{R}$
* Gradient operator $G(x): \mathcal{P} \rightarrow \mathcal{G}$

Note that the product of functions $e$ and $G(x)$ above is point-wise in their common domain $\mathcal{P} = D[\mathbb{R}]$, resulting in the scalar ($\mathbb{R}$) multiplication of vectors in $\mathcal{G}$.

Updating vector $\bm{p}$ to vector $\bm{p} \oplus U(\bm{p})$ in the Parameters Space $\mathcal{P}$ results in updating function $I(\bm{p}): \mathcal{X} \rightarrow \mathbb{R}$ to function $I(\bm{p} \oplus U(\bm{p})): \mathcal{X} \rightarrow \mathbb{R}$ in the Delegator Space $\mathcal{R}$. This is rather convenient since we can view the $\oplus$ operation for the Parameters Space $\mathcal{P}$ as effectively the $\oplus$ operation in the Delegator Space $\mathcal{R}$.

### SGD Update for Linear Function Approximations

\index{function approximation!gradient descent}
\index{function approximation!linear}

In this section, we restrict to linear function approximations, i.e., for all $x \in \mathcal{X}$,
$$f(x, \bm{p}) = \bm{\Phi}(x)^T \cdot \bm{p}$$
where $\bm{p} \in \mathbb{R}^m = \mathcal{P}$ and $\bm{\Phi}: \mathcal{X} \rightarrow \mathbb{R}^m$ represents the feature functions (note: $\bm{\Phi}(x)^T \cdot \bm{p}$ is the usual inner-product in $\mathbb{R}^m$).

Then the gradient function $G: \mathcal{X} \rightarrow (\mathbb{R}^m \rightarrow \mathbb{R}^m)$ can be written as:
$$G(x)(\bm{p}) = \nabla_{\bm{p}} (\bm{\Phi}(x)^T \cdot \bm{p}) = \bm{\Phi}(x)$$
for all $x \in \mathcal{X}$, for all $\bm{p} \in \mathbb{R}^m$.

When SGD-updating vector $\bm{p}$ to vector $\bm{p} \oplus (\alpha * ((y - \bm{\Phi}(x)^T \cdot \bm{p}) * \bm{\Phi}(x)))$ in the Parameters Space $\mathcal{P} = \mathbb{R}^m$, applying the affine map $I: \mathbb{R}^m \rightarrow \mathcal{R}$ correspondingly updates functions in $\mathcal{R}$.  Concretely, a linear function approximation $g: \mathcal{X} \rightarrow \mathbb{R}$ defined as $g(z) = \bm{\Phi}(z)^T \cdot \bm{p}$ for all $z \in \mathcal{X}$ updates correspondingly to the function $g^{(x,y)}: \mathcal{X} \rightarrow \mathbb{R}$ defined as $g^{(x,y)}(z) = \bm{\Phi}(z)^T \cdot \bm{p} + \alpha \cdot (y - \bm{\Phi}(x)^T \cdot \bm{p}) \cdot (\bm{\Phi}(z)^T \cdot \bm{\Phi}(x))$ for all $z \in \mathcal{X}$.

It's useful to note that the change in the evaluation at $z \in \mathcal{X}$, i.e., $g^{(x,y)}(z) - g(z)$, is simply the product of:

* Learning rate $\alpha \in \mathbb{R}^+$
* Prediction Error $y - \bm{\Phi}(x)^T \cdot \bm{p} \in \mathbb{R}$ for the updating data $(x,y) \in \mathcal{X} \times \mathbb{R}$
* Inner-product of the feature vector $\bm{\Phi}(x) \in \mathbb{R}^m$ of the updating input value $x \in \mathcal{X}$ and the feature vector $\bm{\Phi}(z) \in \mathbb{R}^m$ of the evaluation input value $z \in \mathcal{X}$.

\index{function approximation|)}
