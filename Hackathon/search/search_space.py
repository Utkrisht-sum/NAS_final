import random
import copy
from models.architecture_encoding import Architecture


# -------- SAMPLE --------
def sample_architecture():
    depth = random.randint(3, 5)

    hidden_units = [
        random.choice([64, 128, 256])
        for _ in range(depth)
    ]

    activation = random.choice(["relu", "tanh"])
    dropout = round(random.uniform(0.0, 0.5), 2)

    return Architecture(depth, hidden_units, activation, dropout)


# -------- CROSSOVER --------
def crossover(parent1, parent2):
    h1 = parent1.hidden_units
    h2 = parent2.hidden_units

    min_len = min(len(h1), len(h2))
    child_units = []

    for i in range(min_len):
        if random.random() < 0.5:
            child_units.append(copy.deepcopy(h1[i]))
        else:
            child_units.append(copy.deepcopy(h2[i]))

    longer = h1 if len(h1) > len(h2) else h2

    for i in range(min_len, len(longer)):
        if random.random() < 0.5:
            child_units.append(copy.deepcopy(longer[i]))

    activation = random.choice([parent1.activation, parent2.activation])
    dropout = random.choice([parent1.dropout, parent2.dropout])

    return Architecture(
        depth=len(child_units),
        hidden_units=child_units,
        activation=activation,
        dropout=dropout
    )


# -------- MUTATION --------
def mutate_architecture(arch):
    new_arch = copy.deepcopy(arch)

    mutation_type = random.choice(["layers", "activation", "dropout"])

    if mutation_type == "layers":
        if len(new_arch.hidden_units) > 0:
            idx = random.randint(0, len(new_arch.hidden_units) - 1)
            new_arch.hidden_units[idx] = random.choice([16, 32, 64, 128, 256])

        if random.random() < 0.3 and len(new_arch.hidden_units) < 5:
            new_arch.hidden_units.append(random.choice([16, 32, 64, 128, 256]))

        if random.random() < 0.3 and len(new_arch.hidden_units) > 1:
            new_arch.hidden_units.pop(random.randint(0, len(new_arch.hidden_units) - 1))

        new_arch.depth = len(new_arch.hidden_units)

    elif mutation_type == "activation":
        new_arch.activation = random.choice(["relu", "tanh"])

    elif mutation_type == "dropout":
        new_arch.dropout = round(random.uniform(0.0, 0.5), 2)

    return new_arch


# -------- UTILS --------
def print_architecture(arch):
    print(arch)


def architecture_to_string(arch):
    return str(arch)