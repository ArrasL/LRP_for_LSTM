
## Hints on how to apply/extend the code

Typically, in order to apply LRP to your own LSTM model and data, you mainly need to adapt the `__init__` and `set_input` methods of the class `LSTM_bidi`.<br/>
These methods should respectively load your trained model, and define the current input sequence, both in the form of Numpy arrays.

The remaining methods mostly don't need to be changed (except in the few cases listed below), if you are using a standard LSTM.

#### Sanity checks

Two types of sanity checks can be performed:

- You can verify that the LSTM forward pass using the `LSTM_bidi` class is consistent with the forward pass of your trained PyTorch/TensorFlow/Keras/(or any other neural network toolbox) model.

- You can verify that the classifier's prediction score for the LRP *target* class is equal to the sum of the LRP "input" relevances (this includes the relevance of the initial hidden and cell states), in the particular setting where the bias' and stabilizer's share of relevance is redistributed equally onto the lower-layer neurons (i.e. when setting `bias_factor` to 1.0 in the function `lrp_linear`).

To perform these numerical checks, it can be useful to move to float64 precision.

#### Linear output layer

In our implementation the linear output layer has no bias. 

If your model has a final bias, you can incorporate this bias as a parameter into the first two calls to the function `lrp_linear` in the output layer (i.e. lines 216-217 in file `LSTM_bidi.py`), instead of using a vector of zeros as we do.

However, in general, the output layer's bias is not necessary for the prediction task, and you can train a model without a bias in the last linear layer. 

We recommend rather the latter option, since this way, by applying LRP on the output layer, the classifier's prediction score value will be redistributed entirely onto the lower-layer neurons, and no relevance will "leak" into the output layer's bias.

#### LSTM weights - gate ordering

Our code assumes that the LSTM weights have the following ordering: i, g, f, o.<br/>
If your trained LSTM model uses another ordering, say i, f, g, o, then you just need to adapt the gate indices accordingly in our implementation:
```python
idx  = np.hstack((np.arange(0,2*d), np.arange(3*d,4*d))).astype(int) # indices of gates i,f,o together
idx_i, idx_f, idx_g, idx_o = np.arange(0,d), np.arange(d,2*d), np.arange(2*d,3*d), np.arange(3*d,4*d) # indices of gates i,f,g,o separately
```

Moreover, if your trained LSTM model has only one bias, instead of two like in our code, you can safely replace all occurences of both biases (i.e. `self.bxh + self.bhh`) by one LSTM bias.

#### Unidirectional LSTM

To adapt the code to a unidirectional LSTM, you can just remove all references to the `LSTM right encoder`, and adapt the number of connected lower-layer neurons accordingly in the output layer (i.e. change the parameter `bias_nb_units` from `2*d` to `d` in the first call to the function `lrp_linear`, line 216 of file `LSTM_bidi.py`).

