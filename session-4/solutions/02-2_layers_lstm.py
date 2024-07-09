
class BachSynth(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_x, 256, 2, dropout=0.3, batch_first=True)
        self.lin_1 = torch.nn.Linear(256, 256)
        self.dropout = torch.nn.Dropout(0.3)
        self.lin_2 = torch.nn.Linear(256, n_x)
    
    def forward(self, X):
        h_x, c_x = torch.zeros((2, 2, X.shape[0], 256))
        out, h_x = self.lstm(X, (h_x, c_x))
        out = self.lin_1(out)
        out = self.dropout(out)
        return self.lin_2(out)

model2 = BachSynth()
print(model2)

# Training loop
for e in range(n_epochs):
    for i in range(n_batches_per_epochs):
        idx = np.random.choice(len(X_train), size=64, replace=False)
        opt.zero_grad()
        y_pred = model2(X_train[idx])[:, -1]
        l = loss(y_pred, y_train[idx])
        l.backward()
        opt.step()
        if i % 5 == 0:
            print(f"Epoch {e} - Iteration {i} - Loss = {l.item()}\r", end='', flush=True)

# Synthesize data
T_y_generated = 200

prediction_l = [p for p in pattern.detach().numpy()[0]]

for note_index in range(T_y_generated):
    note = torch.nn.functional.softmax(model2(pattern), dim=-1)[0, -1]
    prediction_l.append(note.clone().detach().numpy())
    pattern[0, :-1] = pattern[0, 1:].clone()
    pattern[0, -1] = note

prediction_l = np.array(prediction_l)
play_from_array(prediction_l[:30])
