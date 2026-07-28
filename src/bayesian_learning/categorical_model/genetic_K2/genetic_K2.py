import copy
import random
from typing import Any, Iterable, Sequence, cast
import deap.base as base
import deap.creator as creator
import deap.tools as tools
import pyagrum as gum

def OX2(gen1: list[int], gen2: list[int]) -> tuple[list[int], list[int]]:
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

    c1 = _ox2_single(gen1, gen2)
    c2 = _ox2_single(gen2, gen1)

    gen1[:] = c1
    gen2[:] = c2
    return gen1, gen2


def mutSIM(individual: list[int]) -> tuple[list[int]]:
    size = len(individual)
    if size < 2:
        return (individual,)

    i, j = sorted(random.sample(range(size), 2))
    individual[i : j + 1] = list(reversed(individual[i : j + 1]))
    return (individual,)


def check_de_jong_convergence(
    population: list[list[int]], alpha: float = 0.95, beta: float = 1.0) -> bool:
    if not population:
        return False
    pop_size: int = len(population)
    num_genes: int = len(population[0])
    if pop_size == 0 or num_genes == 0:
        return False
    converged_genes: int = 0

    counts: dict[int, int]
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
    return (max(recent) - min(recent)) <= tol or recent[-1] <= recent[0] + tol


def k2_apply( location: str, attributes: Sequence[str | int],
    max_degree: int = MAX_DEGREE,) -> tuple[gum.BayesNet, float]:
    learner: gum.BNLearner = gum.BNLearner(location)
    node_ids = [
        learner.idFromName(a) if isinstance(a, str) else a for a in attributes
    ]
    learner.useK2(node_ids)
    if max_degree is not None:
        learner.setMaxIndegree(max_degree)
    bn: gum.BayesNet = learner.learnBN()
    nodes_list: list[Any] = list(cast(Iterable[Any], bn.nodes()))
    score: float = sum(learner.score(node) for node in nodes_list)
    return bn, score


def genetic_k2( location: str, pop_size: int = 50, ngen: int = 50,
                cxpb: float = 0.8, mutpb: float = 0.2, tournsize: int = 3,
                max_degree: int = 4, alpha: float = 0.95,
                beta: float = 1.0, patience: int = 10,
                seed: int | None = None) -> tuple[gum.BayesNet, float]:
    """Learns a Bayesian Network structure using a Genetic Algorithm to optimize

    the K2 node ordering, employing OX2 crossover, SIM mutation, and stopping
    criteria based on Larrañaga et al. (1996): 1) Population convergence (De
    Jong: alpha/beta gene convergence) 2) Average population fitness
    stagnation over `patience` generations.
    """
    if seed is not None: random.seed(seed)

    learner: gum.BNLearner = gum.BNLearner(location)
    var_names: list[str] = list(cast(Iterable[str], learner.names()))
    num_vars: int = len(var_names)

    if num_vars == 0:
        raise ValueError(f"No variables found in dataset at '{location}'")

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)  # type: ignore[attr-defined]

    toolbox = base.Toolbox()
    indices_seq: list[int] = list(range(num_vars))
    toolbox.register("indices", random.sample, indices_seq, num_vars)
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, toolbox.indices  # type: ignore[attr-defined]
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # type: ignore[attr-defined]

    fitness_cache: dict[tuple[int, ...], float] = {}

    def evaluate(individual: list[int]) -> tuple[float]:
        key: tuple[int, ...] = tuple(individual)
        if key in fitness_cache:
            return (fitness_cache[key],)

        node_ids: list[int] = list(key)
        learner_inst: gum.BNLearner = gum.BNLearner(location)
        learner_inst.useK2(node_ids)
        learner_inst.setMaxIndegree(max_degree)
        bn = learner_inst.learnBN()
        nodes_list: list[Any] = list(cast(Iterable[Any], bn.nodes()))
        score = sum(learner_inst.score(node) for node in nodes_list)
        fitness_cache[key] = score
        return (score,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", OX2)
    toolbox.register("mutate", mutSIM)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    toolbox.register("clone", copy.deepcopy)

    pop = toolbox.population(n=pop_size)  # type: ignore[attr-defined]
    hof = tools.HallOfFame(1)

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = [toolbox.evaluate(ind) for ind in invalid_ind]  # type: ignore[attr-defined]
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)
    avg_fitness_history: list[float] = []

    for gen in range(1, ngen + 1):
        offspring = toolbox.select(pop, len(pop))  # type: ignore[attr-defined]
        offspring = [toolbox.clone(ind) for ind in offspring]  # type: ignore[attr-defined]

        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                offspring[i - 1], offspring[i] = toolbox.mate(  # type: ignore[attr-defined]
                    offspring[i - 1], offspring[i]
                )
                del offspring[i - 1].fitness.values
                del offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < mutpb:
                offspring[i], = toolbox.mutate(offspring[i])  # type: ignore[attr-defined]
                del offspring[i].fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = [toolbox.evaluate(ind) for ind in invalid_ind]  # type: ignore[attr-defined]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)

        avg_fit = sum(ind.fitness.values[0] for ind in pop) / len(pop)
        avg_fitness_history.append(avg_fit)

        # Stopping Criteria (Larrañaga et al., 1996):
        # 1. Population Convergence (De Jong)
        if check_de_jong_convergence(pop, alpha=alpha, beta=beta):
            break

        # 2. Lack of improvement in average fitness over `patience` generations
        if check_fitness_stagnation(avg_fitness_history, patience=patience):
            break

    best_individual = hof[0]
    best_bn, best_score = k2_apply(
        location, best_individual, max_degree=max_degree
    )
    return best_bn, best_score
