# Step 1
## Target:

1. Get the set-up right
2. Set Transforms
3. Set Data Loader
4. Set test and train loop
5. Set Basic Training  & Test Loop
6. Arranged to reduse the parameter


## Results:
1. Parameters: 11,145
2. Best Training Accuracy: 99.75
3. Best Test Accuracy: 99.31

## Analysis:
1. Even if the model is pushed further, it won't be able to get to 99.4
2. we have seen over-fitting in model, we are changing our model in the next step

# Step 2
## Target:
1) Added GAP
2) Added Dropout to all layer
3) Almost gain good result with less parameter then step_1

## Results:
1) Parameters: 9,434
2) Best Training Accuracy: 98.58
3) Best Test Accuracy: 99.09

## Analysis:
1) The model is pushed further up to 20 EPOCH, but it won't be able to get to 99.4
2) I have seen Under-fitting in model, I am updating the model in the next step with proper location of dropout and max pooling to get batter accuracy

# Step 3
## Target:
1) Adjusted location of Dropout, Batch-norm and max pool
2) Reduce the Dropout value to 0.10
3) Added scheduler
4) In optimizer changed Lr to 0.02 for batter result
5) fix the location of max pool


## Results:
1) Parameters: 7864
Best Training Accuracy: 99.16
Best Test Accuracy: 99.52

## Analysis:
1) With proper adjustment in LR and schduler we got upto 99.52(EPOCH 10) test accuracy, final test accuracy 99.45(EPOCH 15)
2) The model is under fitting.
3) We dont need to use drop out in all layer. 
Need to multiple changes to get proper location to use it.
 