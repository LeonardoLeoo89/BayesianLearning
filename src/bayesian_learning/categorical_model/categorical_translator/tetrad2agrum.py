from typing import Any

class BrokenInvariantException(Exception):
    pass

class FailedCastException(Exception):
    pass

def translate(tetrad: Any) -> Any:
    import pyagrum as gum
    from jpype import JClass
    
    DiscreteVariable = JClass("edu.cmu.tetrad.data.DiscreteVariable")
    
    out = gum.BayesNet()
    node_map: dict[Any, int] = dict()
    name: str
    for node in tetrad.getNodes():
        if not isinstance(node, DiscreteVariable):
            raise FailedCastException("The node is not a DiscreteVariable")
        name = str(node.getName())
        node_map[node] = out.add(gum.LabelizedVariable(name, "", [str(c) for c in node.getCategories()]))

    edges_to_add = []
    for edge in tetrad.getEdges():
        node1 = edge.getNode1()
        node2 = edge.getNode2()
        ep1_name = str(edge.getEndpoint1())
        ep2_name = str(edge.getEndpoint2())
        
        u = node_map[node1]
        v = node_map[node2]
        
        # Priority 0: strict directed (TAIL -> ARROW)
        # Priority 1: partially directed (CIRCLE -> ARROW)
        # Priority 2: undirected or bidirected (CIRCLE -> CIRCLE, ARROW -> ARROW)
        # For priority 1 and 2, direction is ambiguous so we allow trying the reverse direction 
        # if the preferred direction creates a cycle.
        
        if ep1_name == "TAIL" and ep2_name == "ARROW":
            edges_to_add.append((u, v, 0, False))
        elif ep2_name == "TAIL" and ep1_name == "ARROW":
            edges_to_add.append((v, u, 0, False))
        elif ep1_name == "CIRCLE" and ep2_name == "ARROW":
            edges_to_add.append((u, v, 1, True))
        elif ep2_name == "CIRCLE" and ep1_name == "ARROW":
            edges_to_add.append((v, u, 1, True))
        else:
            edges_to_add.append((u, v, 2, True))
            
    # Sort by priority (0 is strict directed, 2 is arbitrary)
    edges_to_add.sort(key=lambda x: x[2])
    
    import pyagrum as gum
    
    for u, v, _, can_reverse in edges_to_add:
        try:
            out.addArc(u, v)
        except Exception:
            if can_reverse:
                try:
                    out.addArc(v, u)
                except Exception:
                    pass

    return out