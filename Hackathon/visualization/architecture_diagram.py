import matplotlib.pyplot as plt


def draw_architecture(arch, input_dim=784, output_dim=10):

    layers = [input_dim] + arch.hidden_units + [output_dim]

    y_positions = range(len(layers))

    plt.figure(figsize=(6,4))

    for i, neurons in enumerate(layers):

        plt.scatter([i]*min(neurons,20), range(min(neurons,20)))

        plt.text(i, 22, f"{neurons}", ha="center")

    for i in range(len(layers)-1):
        plt.plot([i, i+1], [10,10])

    plt.title("AI-Discovered Architecture")

    plt.axis("off")

    plt.savefig("best_architecture.png")

    plt.show()