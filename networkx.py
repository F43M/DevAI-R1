class DiGraph:
    def __init__(self):
        self._adj = {}
    def add_node(self, node, **attrs):
        self._adj.setdefault(node, {})
    def add_edge(self, u, v):
        self.add_node(u)
        self.add_node(v)
        self._adj[u][v] = True
    @property
    def nodes(self):
        return list(self._adj.keys())
    def nodes_with_data(self):
        return [(n, {}) for n in self._adj]
    def successors(self, node):
        return list(self._adj.get(node, {}).keys())
    @property
    def edges(self):
        return [(u, v) for u, targets in self._adj.items() for v in targets]
    def number_of_edges(self):
        return sum(len(t) for t in self._adj.values())

def descendants(graph, source):
    seen = set()
    stack = list(graph.successors(source))
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(graph.successors(node))
    return seen
