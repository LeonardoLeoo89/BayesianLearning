from typing import Any
import pyagrum as gum

class BrokenInvariantException(Exception):
    pass

class FailedCastException(Exception):
    pass

def translate(tetrad: Any) -> gum.DAG:
    """Translation method for tetrad graphs.

    Transforms a tetrad CPDAG/PAG into a pyAgrum DAG by
    orienting non-oriented edges. This is done by topologically
    sorting the directed edges first.

    Args:
        tetrad: The tetrad graph to be translated.

    Returns:
        The graph translated into a pyAgrum DAG.

    Raises:
        BrokenInvariantException: If the directed part of the graph contains a cycle.
        FailedCastException: If the variables aren't all discrete.
    """
    from jpype import JClass
    
    DiscreteVariable = JClass("edu.cmu.tetrad.data.DiscreteVariable")
    
    out: gum.BayesNet = gum.BayesNet()
    node_map: dict[Any, int] = dict()
    name: str
    for node in tetrad.getNodes():
        if not isinstance(node, DiscreteVariable):
            raise FailedCastException("The node is not a DiscreteVariable")
        name = str(node.getName())
        node_map[node] = out.add(gum.LabelizedVariable(name, "", [str(c) for c in node.getCategories()]))

    dag_copy: dict[int, set[int]] = {n : set() for n in node_map.values()}
    ambiguous: set[tuple[int, int]] = set()
    for edge in tetrad.getEdges():
        node1 = edge.getNode1()
        node2 = edge.getNode2()
        ep1_name = str(edge.getEndpoint1())
        ep2_name = str(edge.getEndpoint2())
        
        u = node_map[node1]
        v = node_map[node2]
        
        if ep1_name in {"TAIL", "CIRCLE"} and ep2_name == "ARROW":
            out.addArc(u, v)
            dag_copy[u].add(v)
        elif ep2_name in {"TAIL", "CIRCLE"} and ep1_name == "ARROW":
            out.addArc(v, u)
            dag_copy[v].add(u)
        else:
            ambiguous.add((u, v))

    last_place: int = 0
    topologic_sort: dict[int, int] = dict()
    counters: dict[int, int] = {n : 0 for n in node_map.values()}
    zeroes: set[int] = set()
    current: int
    for tail in dag_copy.keys():
        for head in dag_copy[tail]:
            counters[head] += 1
    for count in counters.keys():
        if counters[count] == 0: zeroes.add(count)
    while zeroes:
        current = zeroes.pop()
        topologic_sort[current] = last_place
        last_place += 1
        for node in dag_copy[current]:
            counters[node] -= 1
            if counters[node] == 0: zeroes.add(node)

    if len(topologic_sort) != len(dag_copy):
        raise BrokenInvariantException("The directed portion of the graph contains a cycle.")

    for u, v in ambiguous:
        if topologic_sort[u] < topologic_sort[v]: out.addArc(u, v)
        else: out.addArc(v, u)

    return out.dag()