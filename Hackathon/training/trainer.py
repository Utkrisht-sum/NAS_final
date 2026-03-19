def train(model, loader, optimizer, loss_fn, device, epochs=3):

    model.train()

    for _ in range(epochs):

        for images, labels in loader:

            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = loss_fn(outputs, labels)

            loss.backward()

            optimizer.step()