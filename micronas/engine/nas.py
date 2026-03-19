import random
import time
import json
import os
import torch
from utils.logger import get_logger
from engine.models import DynamicMLP, DynamicCNN, count_parameters, estimate_memory_mb
from engine.prompt_parser import PromptParser

logger = get_logger("NASEngine")

class NASEngine:
    def __init__(self, metadata, train_loader, val_loader, prompt=""):
        self.metadata = metadata
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.prompt = prompt

        parser = PromptParser()
        self.weights = parser.parse(prompt)

        self.population = []
        self.history = []
        self.pareto_front = []
        self.failures = [] # Failure-aware system

        self.best_model = None
        self.best_config = None

    def _sample_mlp_config(self):
        depth = random.randint(1, 4)
        hidden_layers = [random.choice([16, 32, 64, 128, 256]) for _ in range(depth)]
        return {
            "type": "mlp",
            "hidden_layers": hidden_layers
        }

    def _sample_cnn_config(self):
        # Using Provided Predefined CNN Templates (Small, Medium, Deep) instead of random layers
        templates = [
            {
                "type": "cnn",
                "name": "Small CNN",
                "conv_layers": [{"channels": 32, "kernel_size": 3}, {"channels": 64, "kernel_size": 3}],
                "fc_layers": [128]
            },
            {
                "type": "cnn",
                "name": "Medium CNN",
                "conv_layers": [{"channels": 32, "kernel_size": 3}, {"channels": 64, "kernel_size": 3}, {"channels": 128, "kernel_size": 3}],
                "fc_layers": [256]
            },
            {
                "type": "cnn",
                "name": "Deep CNN",
                "conv_layers": [{"channels": 64, "kernel_size": 3}, {"channels": 128, "kernel_size": 3}, {"channels": 256, "kernel_size": 3}, {"channels": 256, "kernel_size": 3}],
                "fc_layers": [512, 128]
            }
        ]
        return random.choice(templates)

    def _build_model(self, config):
        try:
            dropout_rate = self.weights.get("dropout_rate", 0.2)
            if config["type"] == "mlp":
                model = DynamicMLP(
                    input_size=self.metadata["input_shape"][0],
                    hidden_layers=config["hidden_layers"],
                    num_classes=self.metadata["num_classes"],
                    task=self.metadata["task"],
                    dropout_rate=dropout_rate
                )
            else:
                model = DynamicCNN(
                    input_shape=self.metadata["input_shape"],
                    conv_layers=config["conv_layers"],
                    fc_layers=config["fc_layers"],
                    num_classes=self.metadata["num_classes"],
                    task=self.metadata["task"],
                    dropout_rate=dropout_rate
                )
            return model
        except Exception as e:
            logger.error(f"Failed to build model from config {config}: {e}")
            self.failures.append(config)
            return None

    def _evaluate_fitness(self, model, config, device="cpu"):
        # F(A) = validation_accuracy (Strict Training-Based Fitness)

        # Real Validation Training (Stage 1)
        val_acc = 0.0
        try:
            from engine.trainer import Trainer
            import logging
            old_level = logging.getLogger("Trainer").level
            logging.getLogger("Trainer").setLevel(logging.CRITICAL) # suppress logs

            model.to(device)
            # We enforce 2 epochs of training for the evaluation on the full subset
            proxy_trainer = Trainer(model, self.train_loader, self.val_loader, task=self.metadata["task"])

            # Using standard training loop
            history = proxy_trainer.train(epochs=2, early_stopping_patience=10)

            val_acc = history['val_acc'][-1] if len(history['val_acc']) > 0 else 0.0

            logging.getLogger("Trainer").setLevel(old_level)
            model.to("cpu")
        except Exception as e:
            logger.warning(f"Training evaluation failed: {e}")
            val_acc = 0.0

        # For hackathon rule compliance, fitness is purely based on validation accuracy
        fitness = val_acc

        # Measure params, latency and memory for analytics only
        params = count_parameters(model)

        try:
            model.to(device)
            dummy_input = next(iter(self.train_loader))[0][:4].to(device)
            start = time.time()
            with torch.no_grad():
                model(dummy_input)
            latency = (time.time() - start) * 1000 # ms
        except:
            latency = 999.0
        finally:
            model.to("cpu")

        memory_mb = estimate_memory_mb(model, self.metadata["input_shape"])

        return {
            "fitness": fitness,
            "accuracy_proxy": val_acc / 100.0, # Kept for UI compatibility (e.g. 0.95)
            "params": params,
            "latency_ms": latency,
            "memory_mb": memory_mb,
            "config": config,
            "model": model
        }

    def run_search(self, population_size=10, generations=3, max_params=1e6):
        logger.info(f"Starting NAS | Pop: {population_size}, Gens: {generations}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Running NAS on {device}")

        # Initialize Population
        for _ in range(population_size):
            config = self._sample_mlp_config() if self.metadata["type"] == "tabular" else self._sample_cnn_config()
            model = self._build_model(config)

            if model and count_parameters(model) < max_params:
                eval_data = self._evaluate_fitness(model, config, device)
                self.population.append(eval_data)

        # Handle case where all models exceeded max_params (Fallback)
        if not self.population:
            logger.warning("No models fit within the max_params constraint. Using default fallback.")
            config = self._sample_mlp_config() if self.metadata["type"] == "tabular" else self._sample_cnn_config()
            model = self._build_model(config)
            self.population.append(self._evaluate_fitness(model, config, device))

        # Evolutionary Loop
        for gen in range(generations):
            logger.info(f"--- Generation {gen+1}/{generations} ---")

            # Sort by fitness
            self.population = sorted(self.population, key=lambda x: x["fitness"], reverse=True)
            self.history.append([ind["fitness"] for ind in self.population])

            # Keep top 50%
            parents = self.population[:max(1, population_size//2)]

            # Mutate to fill population
            next_gen = parents.copy()
            while len(next_gen) < population_size:
                parent = random.choice(parents)["config"]
                child_config = self._mutate(parent)

                # Check failure memory
                if child_config in self.failures:
                    logger.info("Avoided known failed architecture.")
                    continue

                child_model = self._build_model(child_config)
                if child_model and count_parameters(child_model) < max_params:
                    eval_data = self._evaluate_fitness(child_model, child_config, device)
                    next_gen.append(eval_data)

            self.population = next_gen

        # Final Selection
        self.population = sorted(self.population, key=lambda x: x["fitness"], reverse=True)
        best = self.population[0]
        self.best_model = best["model"]
        self.best_config = best["config"]

        logger.info(f"NAS Completed. Best Fitness: {best['fitness']:.4f}")
        logger.info(f"Best Config: {self.best_config}")

        self._save_memory(best)

        return self.best_model, self.best_config, self.population

    def _mutate(self, config):
        import copy
        new_config = copy.deepcopy(config)

        if new_config["type"] == "mlp":
            # Add or remove a layer, or change units
            if random.random() < 0.3 and len(new_config["hidden_layers"]) < 5:
                new_config["hidden_layers"].append(random.choice([16, 32, 64]))
            elif random.random() < 0.3 and len(new_config["hidden_layers"]) > 1:
                new_config["hidden_layers"].pop()
            else:
                idx = random.randint(0, len(new_config["hidden_layers"]) - 1)
                new_config["hidden_layers"][idx] = random.choice([16, 32, 64, 128])
        else:
            if random.random() < 0.3 and len(new_config["conv_layers"]) < 5:
                new_config["conv_layers"].append({"channels": random.choice([16,32]), "kernel_size": 3})
            elif random.random() < 0.3 and len(new_config["conv_layers"]) > 1:
                new_config["conv_layers"].pop()
            else:
                idx = random.randint(0, len(new_config["conv_layers"]) - 1)
                new_config["conv_layers"][idx]["channels"] = random.choice([16, 32, 64])

        return new_config

    def _save_memory(self, best_eval):
        os.makedirs("project_output", exist_ok=True)
        memory = {
            "dataset_type": self.metadata["type"],
            "input_shape": self.metadata["input_shape"],
            "best_config": best_eval["config"],
            "performance": {
                "fitness": best_eval["fitness"],
                "params": best_eval["params"],
                "memory_mb": best_eval["memory_mb"]
            }
        }
        with open("project_output/architecture_memory.json", "w") as f:
            json.dump(memory, f, indent=4)
        logger.info("Saved architecture memory to disk.")
