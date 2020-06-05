# Inventory Process

Denote State as $(\alpha, \beta)$ where $\alpha$ is On-Hand and $\beta$ is On-Order. Assume lead time of 1 day. Assume you have space for at most $C$ bicycles in your store, so you decide to follow a policy of ordering the following number of bicycles each evening:
$$max(C - (\alpha + \beta), 0)$$
Assume demand follows a Poisson distribution with Poisson parameter $\lambda$, i.e. demand $i$ for each $i = 0, 1, 2, \ldots$ occurs with probability
$$\frac {e^{-\lambda} \lambda^i} {i!}$$

If current state is $(\alpha, \beta)$, there are only $\alpha + \beta + 1$ possible next states:
$$(\alpha + \beta - i, \max(C - (\alpha + \beta), 0)) \text{ for } i =0, 1, \ldots, \alpha + \beta$$
with transition probabilities governed the the Poisson probabilities of demand as follows:
$$\mathcal{P}((\alpha, \beta), (\alpha + \beta - i, \max(C - (\alpha + \beta), 0))) = \frac {e^{-\lambda} \lambda^i} {i!} \text{ for } 0 \leq i \leq \alpha + \beta - 1$$
$$\mathcal{P}((\alpha, \beta), (0, \max(C - (\alpha + \beta), 0))) = \sum_{j=\alpha+\beta}^{\infty} \frac {e^{-\lambda} \lambda^j} {j!} $$
Note that the On-Hand can be zero from any of demand outcomes greater than or equal to $\alpha + \beta$.

Let overnight holding cost be $h$ per unit of inventory and let daily stockout cost be $p$ per unit of missed demand.
Now let us work out the transition rewards function. When next state's On-Hand is greater than zero, it means all of the day's demand was satisfied with inventory that was available at store-opening ($=\alpha + \beta$), and hence, each of these next states correspond to no stockout cost and only an overnight holding cost of $h \cdot \alpha$. Therefore,
$$\mathcal{R}((\alpha, \beta), (\alpha + \beta - i, \max(C - (\alpha + \beta), 0))) = h \cdot \alpha \text{ for } 0 \leq i \leq \alpha + \beta - 1$$
When next state's On-Hand is equal to zero, there are two possibilities: either (i) the demand was $\alpha + \beta$, meaning all demand was satisifed (so no stockout cost and only overnight holding cost), or (ii) demand was greater than $\alpha + \beta$, in which case there is a stockout cost in addition to overnight holding cost. The stockout cost is an expectation calculation of the number of units of missed demand under the corresponding poisson probabilities of demand exceeding $\alpha + \beta$. Formally, this works out to:
$$\mathcal{R}((\alpha, \beta), (0, \max(C - (\alpha + \beta), 0))) = h \cdot \alpha + p \cdot \sum_{j=\alpha+\beta+1}^{\infty} \frac{e^{-\lambda} \lambda^j} {j!} \cdot (j - (\alpha + \beta))$$ 
