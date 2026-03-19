import matplotlib.pyplot as plt


def plot_accuracy(history):
    generations = [h["generation"] for h in history]
    acc = [h["accuracy"] for h in history]

    plt.figure()
    plt.scatter(generations, acc)
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Generations")
    plt.savefig("accuracy_plot.png")
    plt.close()


def plot_params(history):
    generations = [h["generation"] for h in history]
    params = [h["params"] for h in history]

    plt.figure()
    plt.scatter(generations, params)
    plt.xlabel("Generation")
    plt.ylabel("Parameters")
    plt.title("Model Size over Generations")
    plt.savefig("params_plot.png")
    plt.close()