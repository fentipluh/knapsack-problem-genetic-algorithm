from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

start = time.time()

file = open('3variables.txt')
N = int(file.readline())
maxCapacity = int(file.readline())
optimal_solution = int(file.readline())
data= np.loadtxt('price_weight.txt')

print(optimal_solution)
print(N)
print(maxCapacity)

# КОНСТАНТЫ ГЕНЕТИЧЕСКОГО АЛГОРИТМА
POPULATION_SIZE = 300
P_CROSSOVER = 0.9  # вероятность кроссовера
P_MUTATION = 0.1   # вероятность мутации
MAX_GENERATIONS = 300
HALL_OF_FAME_SIZE =1

np.random.seed(42)
random.seed(42)

toolbox = base.Toolbox()
toolbox.register("zeroOrOne", random.randint, 0, 1)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, N)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

def getValue(individual):
    totalWeight = totalValue = 0
    for i in range(N):
        weight = data[i][1]
        value = data[i][0]
        if totalWeight + weight <= maxCapacity:
            totalWeight += individual[i] * weight
            totalValue += individual[i] * value
    return totalValue

def Value(individual):
    return getValue(individual),

toolbox.register("evaluate", Value)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/N)

population = toolbox.populationCreator(n=POPULATION_SIZE)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("avg", np.mean)

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)
best = hof.items[0]
print("Лучший вектор решений = ", best)
print("Оптимальное решение= ", best.fitness.values[0])

maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

sns.set_style("whitegrid")
plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Generation')
plt.ylabel('Max / Average Fitness')
plt.title('Max and Average fitness over Generations')
plt.show()
end = time.time() - start
print(end)