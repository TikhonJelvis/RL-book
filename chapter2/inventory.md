# Inventory Process

Denote State as $(\alpha, \beta)$ where $\alpha$ is On-Hand and $\beta$ is On-Order. Assume lead time of 1 day. Assume you have space for at most $C$ bicycles in your store, so you decide to follow a policy of ordering the following number of bicycles each evening:
$$max(C - (\alpha + \beta), 0)$$
Assume demand follows a Poisson distribution with Poisson parameter $\lambda$, i.e. demand $i$ for each $i = 0, 1, 2, \ldots$ occurs with probability
$$\frac {e^{-\lambda} \lambda^i} {i!}$$

If current state is $(\alpha, \beta)$, there are only $\alpha + \beta + 1$ possible next states:
$$(\alpha + \beta - i, \max(C - (\alpha + \beta), 0)) \text{ for } i =0, 1, \ldots, \alpha + \beta$$
with transition probabilities:
$$\mathcal{P}((\alpha, \beta), (\alpha + \beta - i, \max(C - (\alpha + \beta), 0))) = \frac {e^{-\lambda} \lambda^i} {i!} \text{ for } 0 \leq i \leq \alpha + \beta - 1$$
$$\mathcal{P}((\alpha, \beta), (0, \max(C - (\alpha + \beta), 0))) = \sum_{j=\alpha+\beta}^{\infty} \frac {e^{-\lambda} \lambda^j} {j!} $$

Let overnight holding cost be $h$ per unit of inventory and let stockout cost be $p$ per unit of missed demand.
Then the transition rewards are:
$$\mathcal{R}((\alpha, \beta), (\alpha + \beta - i, \max(C - (\alpha + \beta), 0))) = h \cdot \alpha \text{ for } 0 \leq i \leq \alpha + \beta - 1$$
$$\mathcal{R}((\alpha, \beta), (0, \max(C - (\alpha + \beta), 0))) = h \cdot \alpha + p \cdot \sum_{j=\alpha+\beta+1}^{\infty} \frac{e^{-\lambda} \lambda^j} {j!} \cdot (j - (\alpha + \beta))$$ 
