"""
User
Is there a Prefix Tree Map data structure where values can be indexed by keys which are lists of strings?
"""
import typing
from typing import TypeVar, Generic, Optional, Sequence

# Define a type variable
K = TypeVar('K')
V = TypeVar('V')


class TrieNode(Generic[K, V]):
    __slots__ = ('value', 'children')

    def __init__(self, value: Optional[V] = None):
        self.value: Optional[V] = value
        self.children: dict[K, TrieNode[K, V]] = {}


def search(root: TrieNode[K, V], sequence: Sequence[K]) -> typing.Optional[TrieNode[K, V]]:
    if not sequence:
        return root
    else:
        first, remaining = sequence[0], sequence[1:]
        if first not in root.children:
            return None
        else:
            return search(root.children[first], remaining)


def search_or_create(root: TrieNode[K, V], sequence: Sequence[K]) -> TrieNode[K, V]:
    if not sequence:
        return root
    else:
        first, remaining = sequence[0], sequence[1:]
        if first not in root.children:
            root.children[first] = TrieNode()
        return search_or_create(root.children[first], remaining)
