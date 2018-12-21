
## Hints on how to apply/extend the code

#### Sanity checks

Two types of sanity checks can be performed.

First, you can verify that the LSTM forward pass computation using the present LRP_for_LSTM code is consistent with the forward pass of your trained PyTorch/TensorFlow/Keras/(or any other toolbox) model.

Second, you can verify that the classifier's prediction score for the LRP target class is equal to the sum of the LRP "input" relevances, when redistributing the bias and the stabilizer value onto the lower-layer units (i.e. when setting `bias_factor` to 1.0 in the function `lrp_linear`).

To perform these numerical checks it can be useful to move to float64 precision.

#### Linear output layer

In our implementation the final linear layer has no bias. If your model has a final bias, you can incorporate it as a parameter in the first two calls to the function `lrp_linear` in the output layer, i.e. lines 213-214 of file `LSTM_bidi.py` (instead of using a vector of zeros as we do).
However, in general, the final layer bias is not needed for prediction, and you can train a model without a bias in the last linear layer. We recommend rather the latter, since this way, with LRP, the classifier's prediction score will be redistributed entirely onto the lower-layer units, and no relevance will "leak" into the output layer bias.

#### Gate ordering in the LSTM weights

Our code assumes that the gates have the following ordering: i, g, f, o.

If your trained model uses another ordering, say i, f, g, o, then you just need to adapt the gate indices accordingly in our implementation:
```python
idx  = np.hstack((np.arange(0,2*d), np.arange(3*d,4*d))).astype(int) # indices of gates i,f,o together
idx_i, idx_f, idx_g, idx_o = np.arange(0,d), np.arange(d,2*d), np.arange(2*d,3*d), np.arange(3*d,4*d) # indices of gates i,f,g,o separately
```
The same way if your LSTM model has only one bias (instead of two like in our code), you can safely replace all occurences of both biases by one bias.

#### Unidirectional LSTM

To adapt the code to unidirectional LSTMs, you can remove all references to the `LSTM right encoder` (and adapt the number of lower-layer units in the output layer, i.e. change the parameter `bias_nb_units` from `2*d` to `d` in the first call to the function `lrp_linear`, line 213 of file `LSTM_bidi.py`).

