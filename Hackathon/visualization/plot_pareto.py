import matplotlib.pyplot as plt


def plot_pareto(history):
    acc = [h["accuracy"] for h in history]
    params = [h["params"] for h in history]

    plt.figure()
    plt.scatter(params, acc)

    # highlight best accuracy point
    max_idx = acc.index(max(acc))
    plt.scatter(params[max_idx], acc[max_idx])
    plt.annotate(
        "Best Acc",
        (params[max_idx], acc[max_idx]),
        textcoords="offset points",
        xytext=(5,5)
    )

    plt.xlabel("Parameters")
    plt.ylabel("Accuracy")
    plt.title("Pareto Frontier (Accuracy vs Params)")

    plt.savefig("pareto.png")
    plt.show()
    plt.close()