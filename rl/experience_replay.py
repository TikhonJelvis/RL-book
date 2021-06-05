from typing import Generic, Iterable, Iterator, List, TypeVar

from rl.distribution import Range

T = TypeVar('T')


class ExperienceReplayMemory(Generic[T]):
    transitions: Iterable[T]
    saved_transitions: List[T]

    def __init__(self, transitions: Iterable[T]):
        self.transitions = transitions
        self.saved_transitions = []

    def replay(self) -> Iterator[T]:
        for transition in self.transitions:
            self.saved_transitions.append(transition)
            yield self.saved_transitions[Range(len(self.saved_transitions)).sample()]

        while True:
            yield self.saved_transitions[Range(len(self.saved_transitions)).sample()]
