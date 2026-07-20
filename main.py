import loader
import cupy as np
import model

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

classifier = model.Model()
classifier.addLayer(inputNo=784,neuronNo=64,layerType="dense", activation="ReLU")
classifier.addLayer(inputNo=64, neuronNo=10, layerType="dense", activation="Softmax")
classifier.setLossFunc()
classifier.setOptimiser()
batchSize = 100
maxEpoch = 600

for epoch in range(maxEpoch):
    xBatch = xTrain[epoch * batchSize: (epoch+1) * batchSize]
    yBatch = yTrain[epoch * batchSize: (epoch+1) * batchSize]
    yBatch = yBatch.astype(int)
    loss, accuracy = classifier.forwardPass(xBatch, yBatch)
    if epoch % 50 == 0:
        print(f"Epoch: {epoch}, loss: {loss}, acc:, {accuracy}")
    classifier.backwardPass(yBatch)
