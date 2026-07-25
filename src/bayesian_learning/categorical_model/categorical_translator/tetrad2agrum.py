import pyagrum as gum
from pytetrad.tools.TetradSearch import TetradSearch
from typing import Any
from jpype import JClass

DiscreteVariable: JClass = JClass("edu.cmu.tetrad.data.DiscreteVariable")

def translate(tetrad: Any) -> gum.BayesNet:
    out: gum.BayesNet = gum.BayesNet()

    for node in tetrad.getNodes():
        pass
    for edge in tetrad.getEdges():
        node1 = edge.getNode1()
        node2 = edge.getNode2()
        if str(edge.getEndpoint1()) == "TAIL":
            pass
    return out
