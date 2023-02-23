# libraries
from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite
from rl.returns import returns


def mc_pred(traces, gamma, tolerace):
    means = {}
    counts = {}
    for episode in (returns(trace, gamma, tolerace) for trace in traces):
        for step in episode:
            counts[step.state] = counts.get(step.state, 0) + 1
            means[step.state] = means.get(
                step.state, 0) + (step.return_ - means.get(step.state, 0)) / counts[step.state]
        yield means


def td_pred(traces, gamma, tolerance, v_0):

    means = {}
    counts = {}
    for episode in enumerate(returns(trace, gamma, tolerance) for trace in traces):
        for step in episode:
            counts[step.state] = counts.get(step.state, 0) + 1
            observed_val = step.reward + γ * \
                means.get(step.next_state, v_0)
            means[step.state] = means.get(
                step.state, 0) + (observed_val - means.get(step.state, 0)) / counts[step.state]
        yield means


if __name__ == '__main__':

    gamma = 0.75

    # Create MRP
    basic_mrp = SimpleInventoryMRPFinite(5, 1.0, 2.5, 2.5)
    traces = basic_mrp.reward_traces(
        rl.distribution.Choose(basic_mrp.non_terminal_states))
    print("Simple Inventory MRP")
    basic_mrp.display_value_function(gamma=gamma)

    # Monte Carlo prediction

    print("Monte Carlo prediction")
    mc = mc_pred(traces, gamma)
    last_elem = None
    for i, elem in enumerate(mc):
        if i >= 1000:
            break
        last_elem = elem
    print(last_elem)

    # TD prediction

    print("TD prediction")
    td = td_pred(traces, gamma, 1e-6, 0)
    last_elem_td = None
    for i, elem in enumerate(mc):
        if i >= 1000:
            break
        last_elem_td = elem
    print(last_elem_td)
