## Target:

1) Utilized Layers:
   - Convolution
   - 1x1 Kernel
   - 3x3 Kernel
   - Batch Normalization
   - Dropout
   - Softmax Activation
   - Learning Rate

2) Parameter Constraint:
   - Ensured model parameter count is less than 20k.

3) Total Epochs:
   - Trained the model for a total of 20 epochs.

## Results:
- Parameters: 16,196
- Best Training Accuracy: 99.19%
- Best Test Accuracy: 99.44%

## Analysis and Additional Considerations:
1) model is Underfitting
2) Learning Rate Scheduler can help in stabilizing training and test
3) Tried different location of Batch-Norm and Dropout location for batter result