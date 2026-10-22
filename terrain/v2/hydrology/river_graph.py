"""River reaches as a graph: outlets, order, and elevations that descend downstream."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RiverNode:
    id: int
    y: int
    x: int
    elevation_m: float
    discharge: float
    downstream_id: int | None
    strahler: int = 1


@dataclass
class RiverGraph:
    nodes: list = field(default_factory=list)
    complete: bool = True

    def node_at(self, y, x):
        for node in self.nodes:
            if node.y == y and node.x == x:
                return node
        return None


def _strahler(orders):
    if not orders:
        return 1
    top = max(orders)
    if sum(order == top for order in orders) >= 2:
        return top + 1
    return top


def build_graph(elevation, receivers, accumulation, threshold):
    """One node per cell whose drainage exceeds `threshold`."""
    height = np.asarray(elevation, np.float64)
    rec = np.asarray(receivers, np.int32)
    acc = np.asarray(accumulation, np.float64)
    rows, cols = height.shape
    channel = acc >= float(threshold)
    flat_of = -np.ones(rows * cols, np.int32)
    nodes = []
    for y, x in zip(*np.nonzero(channel)):
        node_id = len(nodes)
        flat_of[y * cols + x] = node_id
        nodes.append(RiverNode(
            id=node_id, y=int(y), x=int(x),
            elevation_m=float(height[y, x]),
            discharge=float(acc[y, x]),
            downstream_id=None,
        ))
    upstream = [[] for _ in nodes]
    for node in nodes:
        flat = node.y * cols + node.x
        dest = int(rec[node.y, node.x])
        if dest < 0:
            continue
        downstream = int(flat_of[dest])
        if downstream < 0:
            continue
        node.downstream_id = downstream
        upstream[downstream].append(node.id)

    indeg = [len(ups) for ups in upstream]
    queue = [i for i, degree in enumerate(indeg) if degree == 0]
    head = 0
    while head < len(queue):
        current = queue[head]
        head += 1
        nodes[current].strahler = _strahler(
            [nodes[item].strahler for item in upstream[current]])
        dest = nodes[current].downstream_id
        if dest is None:
            continue
        indeg[dest] -= 1
        if indeg[dest] == 0:
            queue.append(dest)
    return RiverGraph(nodes=nodes, complete=len(queue) == len(nodes))


def enforce_descending(graph, min_drop=0.1):
    """Raise an upstream node that sits below the reach it flows into."""
    indeg = [0] * len(graph.nodes)
    for node in graph.nodes:
        if node.downstream_id is not None:
            indeg[node.downstream_id] += 1
    queue = [i for i, degree in enumerate(indeg) if degree == 0]
    head = 0
    while head < len(queue):
        head += 1
        dest = graph.nodes[queue[head - 1]].downstream_id
        if dest is not None and indeg[dest] > 0:
            indeg[dest] -= 1
            if indeg[dest] == 0:
                queue.append(dest)
    for index in reversed(queue):
        node = graph.nodes[index]
        if node.downstream_id is None:
            continue
        floor = graph.nodes[node.downstream_id].elevation_m + float(min_drop)
        if node.elevation_m < floor:
            node.elevation_m = floor
    return graph
