import loader
import cupy as np
import model
import time

#loading training and testing data
def loadData():
    dataloader = loader.MnistDataloader(training_images_filepath="train-images.idx3-ubyte", training_labels_filepath="train-labels.idx1-ubyte", test_images_filepath="t10k-images.idx3-ubyte", test_labels_filepath="t10k-labels.idx1-ubyte")
    (xTrainRaw, yTrainRaw), (xTestRaw, yTestRaw) = dataloader.load_data()
    print("Data Loaded!")
    return (xTrainRaw, yTrainRaw), (xTestRaw, yTestRaw)

def dataCleaning(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw):
    xTrain = np.array(xTrainRaw)
    yTrain = np.array(yTrainRaw)
    xTest = np.array(xTestRaw)
    yTest = np.array(yTestRaw)

    xTrain = (xTrain.astype(np.float32) - 128.0) / 128.0
    xTest = (xTest.astype(np.float32) - 128.0) / 128.0

    xTrain = xTrain.reshape(xTrain.shape[0], -1)
    xTest = xTest.reshape(xTest.shape[0], -1)

    print(f"Shape: {xTrain.shape}")
    return (xTrain, yTrain), (xTest, yTest)

(xTrainRaw, yTrainRaw), (xTestRaw, yTestRaw) = loadData()
(xTrain, yTrain), (xTest, yTest) = dataCleaning(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw)

start = time.time()
classifier = model.Model()
classifier.setLossFunc()
classifier.setOptimiser()

xVal = xTrain[50000:]
yVal = yTrain[50000:]
xTrain = xTrain[:50000]
yTrain = yTrain[:50000]

classifier.sizeOptimiser(784, 10, xTrain, yTrain, xVal, yVal)

#default hyper parameters for Adam optimiser
hyperParams = {
    "learningRate": 0.001, 
    "decay": 0.0,
    "epsilon": 1e-7,
    "beta1": 0.9,
    "beta2": 0.999
}

classifier.hyperParamOptimiser(xTrain, yTrain, xVal, yVal, hyperParams, "Adam")

print("Main training cycle")
for epoch in range (500):
    loss, acc = classifier.cycle(xTrain, yTrain)
    if epoch % 5 == 0:
        print(f"Epoch: {epoch}, loss: {loss}, acc: {acc}")

loss, acc = classifier.forwardPass(xTest, yTest)
print(f"Test: Loss - {loss}, Acc - {acc}")
print(f"Total Time: {time.time() - start}")