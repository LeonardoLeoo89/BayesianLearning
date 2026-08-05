import copy
import random
from typing import Any, Iterable, Sequence, cast

from sympy.series import sequences


def ox2(gen1: list[int], gen2: list[int]) -> tuple[list[int], list[int]]:
    size: int = len(gen1)
    if size < 2:
        return gen1, gen2

    def _ox2_single(p1: list[int], p2: list[int]) -> list[int]:
        k: int = random.randint(1, size - 1)
        idx: int = 0
        selected_positions: set[int] = set(random.sample(range(size), k))
        selected_elements: set[int] = {p2[pos] for pos in selected_positions}
        ordered_selected: list[int] = [val for val in p2 if val in selected_elements]
        child: list[int] = [-1] * size

        for i, val in enumerate(p1):
            if val not in selected_elements:
                child[i] = val

        for i in range(size):
            if child[i] == -1:
                child[i] = ordered_selected[idx]
                idx += 1
        return child

    c1: list[int] = _ox2_single(gen1, gen2)
    c2: list[int] = _ox2_single(gen2, gen1)

    gen1[:] = c1
    gen2[:] = c2
    return gen1, gen2


def mut_sim(individual: list[int]) -> tuple[list[int]]:
    size: int = len(individual)
    if size < 2:
        return (individual,)

    i, j = sorted(random.sample(range(size), 2))
    individual[i : j + 1] = list(reversed(individual[i : j + 1]))
    return (individual,)


def check_de_jong_convergence(population: list[list[int]], alpha: float = 0.95,
                              beta: float = 1.0) -> bool:
    if not population:
        return False
    pop_size: int = len(population)
    num_genes: int = len(population[0])
    if pop_size == 0 or num_genes == 0:
        return False
    converged_genes: int = 0
    counts: dict[int, int]
    val: int

    for pos in range(num_genes):
        counts = {}
        for ind in population:
            val = ind[pos]
            counts[val] = counts.get(val, 0) + 1
        if (max(counts.values()) / pop_size) >= alpha:
            converged_genes += 1

    return (converged_genes / num_genes) >= beta


def check_fitness_stagnation( avg_fitness_history: list[float], patience: int = 10,
                              tol: float = 1e-6 ) -> bool:
    if len(avg_fitness_history) < patience + 1: return False
    recent: list[float] = avg_fitness_history[-(patience + 1) :]
    return (max(recent) - min(recent)) <= tol or recent[-1] - recent[0] <= tol


def k2_apply( location: str, attributes: Sequence[Any],
    max_degree: int = 4) -> gum.BayesNet:
    import pyagrum as gum
    learner: gum.BNLearner = gum.BNLearner(location)
    learner.useSmoothingPrior(1.0)
    node_ids: list[int] = [
        learner.idFromName(a) if isinstance(a, str) else a for a in attributes
    ]
    learner.useK2(node_ids)
    if max_degree is not None:
        learner.setMaxIndegree(max_degree)
    bn: gum.BayesNet = learner.learnBN()
    return bn


import pyagrum as gum
def genetic_k2( location: str, pop_size: int = 50, ngen: int = 50,
                cxpb: float = 0.8, mutpb: float = 0.2, tournsize: int = 3,
                max_degree: int = 4, alpha: float = 0.95,
                beta: float = 1.0, patience: int = 10,
                seed: int | None = None) -> gum.BayesNet:
    import deap.base as base
    import deap.creator as _creator
    import deap.tools as tools
    creator: Any = _creator

    if seed is not None: random.seed(seed)

    learner = gum.BNLearner(location)
    learner.useSmoothingPrior(1.0)
    var_names: list[str] = list(cast(Iterable[str], learner.names()))
    num_vars: int = len(var_names)
    if num_vars == 0:
        raise ValueError(f"No variables found in dataset at '{location}'")

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox: Any = base.Toolbox()
    indices_seq: list[int] = list(range(num_vars))
    toolbox.register("indices", random.sample, indices_seq, num_vars)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    fitness_cache: dict[tuple[int, ...], float] = {}

    def evaluate(individual: list[int]) -> tuple[float]:
        key: tuple[int, ...] = tuple(individual)
        if key in fitness_cache:
            return (fitness_cache[key],)

        node_ids: list[int] = list(key)
        learner_inst: gum.BNLearner = gum.BNLearner(location)
        learner_inst.useSmoothingPrior(1.0)
        learner_inst.useK2(node_ids)
        learner_inst.setMaxIndegree(max_degree)
        bn: gum.BayesNet = learner_inst.learnBN()
        nodes_list: list[Any] = list(cast(Iterable[Any], bn.nodes()))
        score: float = sum(learner_inst.score(node) for node in nodes_list)
        fitness_cache[key] = score
        return (score,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", ox2)
    toolbox.register("mutate", mut_sim)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    toolbox.register("clone", copy.deepcopy)

    pop: list[Any] = toolbox.population(pop_size)
    hof: tools.HallOfFame = tools.HallOfFame(1)

    invalid_ind: list[Any] = [ind for ind in pop if not ind.fitness.valid]
    fitnesses: list[Any] = [toolbox.evaluate(ind) for ind in invalid_ind]
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)
    avg_fitness_history: list[float] = []
    offspring: list[Any]
    invalid_ind: list[Any]
    avg_fit: float

    for gen in range(1, ngen + 1):
        offspring = [toolbox.clone(ind) for ind in toolbox.select(pop, len(pop))]

        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                offspring[i - 1], offspring[i] = toolbox.mate(
                    offspring[i - 1], offspring[i]
                )
                del offspring[i - 1].fitness.values
                del offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < mutpb:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = [toolbox.evaluate(ind) for ind in invalid_ind]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)

        avg_fit = sum(ind.fitness.values[0] for ind in pop) / len(pop)
        avg_fitness_history.append(avg_fit)

        if check_de_jong_convergence(pop, alpha, beta):
            break

        if check_fitness_stagnation(avg_fitness_history, patience):
            break

    best_individual: Sequence[Any] = hof[0]
    best_bn: gum.BayesNet = k2_apply(
        location, best_individual, max_degree
    )
    return best_bn
