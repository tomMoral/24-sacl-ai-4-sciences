
def train(model, optimizer, criterion, n_epochs=10):

    # Loop over epochs
    for e in range(n_epochs):

        # Loop over data batches
        running_loss = 0
        for xb, yb in train_loader:

            # Training pass
            # --------------
            # Zero gradients
            optimizer.zero_grad()
            # Forward pass
            prob = model(xb)
            # Compute loss
            loss = criterion(prob, yb)
            # Backwards pass
            loss.backward()
            # Gradient step
            optimizer.step()

            running_loss += loss.item()
        training_loss = running_loss / len(train_loader)

        val_loss, val_accuracy = evaluate_model(model)
        print("Epoch number", e+1)
        print("------------")
        print(f"Training loss: {training_loss}")
        print(f"Val loss: {val_loss}")
        print(f"Val Accuracy: {val_accuracy}\n")
