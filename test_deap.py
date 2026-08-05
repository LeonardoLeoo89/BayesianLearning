from deap import base, creator, tools

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("select", tools.selTournament, tournsize=3)

pop = [creator.Individual([1, 2]), creator.Individual([3, 4])]
for ind in pop:
    ind.fitness.values = (1.0,)

print("pop type:", type(pop))
print("len pop:", len(pop))
toolbox.select(pop, len(pop))
