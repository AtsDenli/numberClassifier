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
classifier.setLossFunc()
classifier.setOptimiser()

xVal = xTrain[50000:]
yVal = yTrain[50000:]
xTrain = xTrain[:50000]
yTrain = yTrain[:50000]

classifier.sizeOptimiser(784, 10, xTrain, yTrain, xVal, yVal)
classifier.saveModel()

loadNew = model.Model()
loadNew.loadModel("NNParameters.pkl", "NNMetaData.pkl")
loss, acc = loadNew.cycle(xTrain, yTrain)
print(f"{loss}, {acc}")