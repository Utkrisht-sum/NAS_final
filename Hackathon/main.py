import torch
import argparse
from dataset.dataset_loader import load_custom_dataset
from search.evolutionary_search import run_evolution
import models.model_builder as mb
from training.trainer import train
from training.evaluator import evaluate
from utils.predict import predict_image


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--predict", type=str, default=None)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- PREDICT MODE ----------------
if args.predict:
    import json
    from models.architecture_encoding import Architecture

    # load dataset ONLY for classes
    _, _, output_dim, classes = load_custom_dataset(args.dataset)

    with open("best_arch.json", "r") as f:
        data = json.load(f)

    best_arch = Architecture(
        depth=data["depth"],
        hidden_units=data["hidden_units"],
        activation=data["activation"],
        dropout=data["dropout"]
    )

    model = mb.CNNModel(
        mb.convert_architecture(best_arch),
        num_classes=output_dim
    ).to(device)

    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    pred = predict_image(model, args.predict, classes, device)
    print(f"\nPrediction for {args.predict}: {pred}")

    exit()

# ---------------- LOAD DATASET ----------------
train_loader, test_loader, output_dim, classes = load_custom_dataset(args.dataset)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Classes:", classes)


# ---------------- NAS SEARCH ----------------
best, history = run_evolution(
    train_loader,
    test_loader,
    output_dim,
    generations=1,
    population_size=3,
    device=device,
    mode="training"
)

best_arch = best[3]

print("\nBest Architecture:", best_arch)


# ---------------- FINAL TRAINING ----------------
model = mb.CNNModel(
    mb.convert_architecture(best_arch),
    num_classes=output_dim
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

train(model, train_loader, optimizer, loss_fn, device, epochs=60)

acc = evaluate(model, test_loader, device)

print(f"\nFinal Accuracy: {acc:.4f}")


# ---------------- SAVE ----------------
torch.save(model.state_dict(), "best_model.pth")
print("Model saved as best_model.pth")


import json

with open("best_arch.json", "w") as f:
    json.dump({
        "depth": best_arch.depth,
        "hidden_units": best_arch.hidden_units,
        "activation": best_arch.activation,
        "dropout": best_arch.dropout
    }, f)

print("Architecture saved as best_arch.json")


