from rl.distribution import Distribution, Constant, Gaussian, Choose, SampledDistribution
from itertools import product
from collections import defaultdict
import operator
from typing import Mapping, Iterator, TypeVar, Tuple, Dict, Iterable, Generic

import numpy as np

from rl.distribution import Categorical, Choose
from rl.markov_process import NonTerminal, State, Terminal
from rl.markov_decision_process import (MarkovDecisionProcess, FiniteMarkovDecisionProcess, FiniteMarkovRewardProcess)
from rl.policy import FinitePolicy, FiniteDeterministicPolicy
from rl.approximate_dynamic_programming import ValueFunctionApprox, \
    QValueFunctionApprox, NTStateDistribution, extended_vf

import random 

from dataclasses import dataclass
from rl import dynamic_programming

from rl.approximate_dynamic_programming import QValueFunctionApprox
from rl.td import PolicyFromQType, epsilon_greedy_action


S = TypeVar('S')
A = TypeVar('A')


class TabularQValueFunctionApprox(Generic[S, A]):
    '''
    A basic implementation of a tabular function approximation with constant learning rate of 0.1
    also tracks the number of updates per state
    You should use this class in your implementation
    '''
    
    def __init__(self):
        self.counts: Mapping[Tuple[NonTerminal[S], A], int] = defaultdict(int)
        self.values: Mapping[Tuple[NonTerminal[S], A], float] = defaultdict(float)
    
    def update(self, k: Tuple[NonTerminal[S], A], tgt):
        alpha = 0.1
        self.values[k] = (1 - alpha) * self.values[k] + tgt * alpha
        self.counts[k] += 1
    
    def __call__(self, x_value: Tuple[NonTerminal[S], A]) -> float:
        return self.values[x_value]



def double_q_learning(
    mdp: MarkovDecisionProcess[S, A],
    states: NTStateDistribution[S],
    γ: float
) -> Iterator[TabularQValueFunctionApprox[S, A]]:
    '''
    Implement the double q-learning algorithm as outlined in the question
    '''
    ##### Your Code HERE #########
    
    ##### End Your Code HERE #########
    
            
def q_learning(
    mdp: MarkovDecisionProcess[S, A],
    states: NTStateDistribution[S],
    γ: float
) -> Iterator[TabularQValueFunctionApprox[S, A]]:
    '''
    Implement the standard q-learning algorithm as outlined in the question
    '''
    ##### Your Code HERE #########
    pass
    ##### End Your Code HERE #########



@dataclass(frozen=True)
class P1State:
    '''
    Add any data and functionality you need from your state
    '''
    ##### Your Code HERE #########
    
    ##### End Your Code HERE #########
    

class P1MDP(MarkovDecisionProcess[P1State, str]):
    
    def __init__(self, n):
        self.n = n
        
        
    def actions(self, state: NonTerminal[P1State]) -> Iterable[str]:
        '''
        return the actions available from: state
        '''
        ##### Your Code HERE #########
        pass
        ##### End Your Code HERE #########
    
    def step(
        self,
        state: NonTerminal[P1State],
        action: str
    ) -> Distribution[Tuple[State[P1State], float]]:
        '''
        return the distribution of next states conditioned on: (state, action)
        '''
        ##### Your Code HERE #########
        pass
        ##### End Your Code HERE #########
