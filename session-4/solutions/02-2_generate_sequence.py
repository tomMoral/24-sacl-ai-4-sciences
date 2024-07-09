T_y_generated = 200

prediction_l = [p for p in pattern.detach().numpy()[0]]

# generate T_y_generated notes
for note_index in range(T_y_generated):
    #######################
    # TODO

    note = torch.nn.functional.softmax(model(pattern), dim=-1)[0, -1]
    prediction_l.append(note.clone().detach().numpy())
    pattern[0, :-1] = pattern[0, 1:].clone()
    pattern[0, -1] = note

    # END TODO
    #######################

prediction_l = np.array(prediction_l)
prediction_l.shape
