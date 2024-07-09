sequence_length = 20

X_train_list = []
y_train_list = []

##########################
# TODO

for X_ohe in X_list:
    for t in range(X_ohe.shape[0] - sequence_length):
        X_train_list.append(X_ohe[t:t + sequence_length, :])
        y_train_list.append(X_ohe[t + sequence_length, :].argmax())

# END TODO
##########################

X_train = np.asarray(X_train_list)
y_train = np.asarray(y_train_list)

print("X_train.shape:", X_train.shape)
print("y_train.shape:", y_train.shape)
