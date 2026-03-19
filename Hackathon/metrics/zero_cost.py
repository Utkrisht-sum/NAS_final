import torch
import torch.nn as nn

def zero_cost_score(model, loader, device):

    model.train()

    loss_fn = nn.CrossEntropyLoss()

    for images, labels in loader:

        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = loss_fn(outputs, labels)

        loss.backward()

        score = 0

        for p in model.parameters():
            if p.grad is not None:
                score += torch.sum(torch.abs(p.grad)).item()

        return score