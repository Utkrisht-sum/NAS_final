import time
import torch

from dataset.dataset_loader import load_dataset
from search.evolutionary_search import run_evolution


train_loader, test_loader, input_dim, output_dim = load_dataset("mnist", 64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("\nRunning STANDARD NAS (training models)...")

start = time.time()

run_evolution(
    train_loader,
    test_loader,
    input_dim,
    output_dim,
    generations=2,
    population_size=4,
    device=device,
    mode="training"
)

end = time.time()

print("Standard NAS Time:", end - start, "seconds")


print("\nRunning OPTIMIZED NAS (zero-cost proxy)...")

start = time.time()

run_evolution(
    train_loader,
    test_loader,
    input_dim,
    output_dim,
    generations=2,
    population_size=4,
    device=device,
    mode="zero_cost"
)

end = time.time()

print("Optimized NAS Time:", end - start, "seconds")