import dis

import pyagrum as gum
from pytetrad.tools.TetradSearch import TetradSearch
from typing import Any
from jpype import JClass

DiscreteVariable: JClass = JClass("edu.cmu.tetrad.data.DiscreteVariable")

class BrokenInvariantException(Exception):
    pass

class FailedCastException(Exception):
    pass

def translate(tetrad: Any) -> gum.BayesNet:
    out: gum.BayesNet = gum.BayesNet()
    map: dict[Any, int] = dict()
    name: str
    for node in tetrad.getNodes():
        if not isinstance(node, DiscreteVariable):
            raise FailedCastException("The node is not a DiscreteVariable")
        name = node.getName()
        map[node] = out.add(gum.LabelizedVariable(name, list(node.getCategories())))

    for edge in tetrad.getEdges():
        node1 = edge.getNode1()
        node2 = edge.getNode2()
        if str(edge.getEndpoint1()) == "ARROW":
            out.addArc(map[node2], map[node1])
        elif str(edge.getEndpoint2()) == "ARROW":
            out.addArc(map[node1], map[node2])
        else:
            raise BrokenInvariantException("The tetrad graph is not a DAG")

    return out