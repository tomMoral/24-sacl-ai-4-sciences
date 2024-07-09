# Step 1: Define training function that takes in hyperparameter
# values from `wandb.config` and uses them to train a model and return metric
def main():

    run = wandb.init()

    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
        nn.LogSoftmax(dim=1)
    )
    optimizer = optim.SGD(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)
    criterion = nn.NLLLoss()

    train(model, optimizer, criterion, verbose=False, use_wandb=True)

# Step 2: Define sweep config
sweep_configuration = {
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'method': 'random',
    'parameters': {
        'lr': {'max': 0.1, 'min': 0.0001},
        'weight_decay': { "values": [0, 1e-5, 1e-4]}
    }
}

# Step 3: Initialize sweep by passing in config
sweep_id = wandb.sweep(sweep=sweep_configuration, project='MLP_covertype')
wandb.agent(sweep_id, function=main, count=6)
