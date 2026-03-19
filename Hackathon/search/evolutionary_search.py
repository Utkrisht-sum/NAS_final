import os
print("RUNNING FILE:", os.path.abspath(__file__))
import random
import torch
import torch.nn as nn
import torch.optim as optim

from search.search_space import sample_architecture, mutate_architecture
import models.model_builder as mb

from metrics.zero_cost import zero_cost_score
from metrics.compute_cost import compute_parameters
from metrics.fitness import fitness

from training.trainer import train
from training.evaluator import evaluate
print("MODEL BUILDER FILE:", mb.__file__)


def run_evolution(
    train_loader,
    test_loader,
    output_dim,
    generations=5,
    population_size=10,
    device="cpu",
    mode="training"
):

    population = []
    history = []

    best_overall = None

    for generation in range(generations):

        print(f"\n=== Generation {generation+1} ===")

        candidates = []

        # -------- INIT FIRST GEN RANDOM --------
        if generation == 0:
            population = [
                (None, None, None, sample_architecture())
                for _ in range(population_size)
            ]

        # -------- EVALUATE POPULATION --------
        for i in range(population_size):

            arch = population[i][3]

            model = mb.CNNModel(
                mb.convert_architecture(arch),
                num_classes=output_dim
            ).to(device)

            params = compute_parameters(model)

            # ---------- REAL TRAINING NAS ----------
            if mode == "training":

                optimizer = optim.Adam(model.parameters(), lr=0.001)
                loss_fn = nn.CrossEntropyLoss()

                train(
                    model,
                    train_loader,
                    optimizer,
                    loss_fn,
                    device,
                    epochs=8
                )

                acc = evaluate(model, test_loader, device)

                score = fitness(acc, params)

                print(
                    f"[Gen {generation+1} | Model {i+1}] "
                    f"Arch: {arch} | Acc: {acc:.4f} | Params: {params}"
                )

            # ---------- ZERO-COST NAS ----------
            else:

                score = zero_cost_score(model, train_loader, device)
                acc = score / max(1, params)

                print(
                    f"[Gen {generation+1} | Model {i+1}] "
                    f"Arch: {arch} | ProxyScore: {score:.2f} | Params: {params}"
                )

            history.append({
                "generation": generation,
                "accuracy": acc,
                "params": params
            })

            candidates.append((score, acc, params, arch))

        # -------- SORT BY FITNESS --------
        candidates.sort(key=lambda x: x[0], reverse=True)

        best = candidates[0]

        print("\nBest architecture this generation:")
        print(best)

        if best_overall is None or best[0] > best_overall[0]:
            best_overall = best

        # -------- SELECTION (TOP HALF) --------
        survivors = candidates[: population_size // 2]

        # -------- CREATE NEXT GEN --------
        new_population_archs = [arch for (_, _, _, arch) in survivors]

        while len(new_population_archs) < population_size:
            parent = random.choice(new_population_archs)
            child = mutate_architecture(parent)
            new_population_archs.append(child)

        # store as same tuple format
        population = [(None, None, None, arch) for arch in new_population_archs]

    print("\n=== NAS COMPLETE ===")
    print("Best architecture overall:")
    print(best_overall)

    return best_overall, history