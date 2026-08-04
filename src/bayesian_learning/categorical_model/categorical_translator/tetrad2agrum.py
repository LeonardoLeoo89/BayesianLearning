from typing import Any

class BrokenInvariantException(Exception):
    pass

class FailedCastException(Exception):
    pass

def translate(tetrad: Any) -> Any:
    import pyagrum as gum
    from jpype import JClass
    
    discrete_variable = JClass("edu.cmu.tetrad.data.discrete_variable")
    
    out = gum.BayesNet()
    node_map: dict[Any, int] = dict()
    name: str
    for node in tetrad.getNodes():
        if not isinstance(node, discrete_variable):
            raise FailedCastException("The node is not a discrete_variable")
        name = node.getName()
        node_map[node] = out.add(gum.LabelizedVariable(name, "", list(node.getCategories())))

    for edge in tetrad.getEdges():
        node1 = edge.getNode1()
        node2 = edge.getNode2()
        ep1_is_arrow = str(edge.getEndpoint1()) == "ARROW"
        ep2_is_arrow = str(edge.getEndpoint2()) == "ARROW"
        if ep1_is_arrow and not ep2_is_arrow:
            out.addArc(node_map[node2], node_map[node1])
        elif ep2_is_arrow and not ep1_is_arrow:
            out.addArc(node_map[node1], node_map[node2])
        else:
            raise BrokenInvariantException("The tetrad graph is not a DAG")

    return out