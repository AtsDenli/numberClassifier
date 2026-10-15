import cupy as np
import pickle as pkl

class Layer_Dense:
    #initialising a Dense Layer
    def __init__(self, inputNo, neuronNo, weightRegL1=0, weightRegL2=5e-4, biasRegL1=0, biasRegL2=5e-4):
        self.inputNo = inputNo
        self.neuronNo = neuronNo
        self.weights = 0.01 * np.random.randn(inputNo, neuronNo)
        self.biases = np.zeros((1, neuronNo))
        self.weightRegL1 = weightRegL1
        self.weightRegL2 = weightRegL2
        self.biasRegL1 = biasRegL1
        self.biasRegL2 = biasRegL2 
        self.type = "dense"

    #forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    #backward pass
    def backward(self, dvalues):
        #dvalues are the gradients w.r.t inputs from the next layer
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.weightRegL1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weightRegL1 * dL1

        if self.weightRegL2 > 0:
            self.dweights += 2 * self.weightRegL2 * self.weights

        if self.biasRegL1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.biasRegL1 * dL1

        if self.biasRegL2 > 0:
            self.dbiases += 2 * self.biasRegL2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)

        
class Layer_Dropout():
    def __init__(self, rate, inputNo):
        #the input is the rate of dropout, but we deal with the rate of success, so we invert it
        self.rate = 1 - rate
        self.neuronNo = inputNo
        self.inputNo = inputNo
        self.type = "dropout"
    
    def forward(self, inputs):
        self.inputs = inputs
        #the mask of the things we will keep. whats randomly assigned as a 0 here will deactivate the related neuron
        self.binaryMask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binaryMask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binaryMask

class Activation_ReLU():
    def __init__(self):
        self.type = "ReLU"

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax():
    def __init__(self):
        self.type = "Softmax"

    def forward(self, inputs):
        self.inputs = inputs
        exponentials = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
        self.output = exponentials/np.sum(exponentials, axis=1, keepdims=True)

    def backward(self, dvalues):
        #unintialised array
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1,1)
            #single_output.T is a standard result in matrix calculus. It is the application of chain rule here
            jacobianMatrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobianMatrix, single_dvalues)

class Activation_Sigmoid():
    def __init__(self):
        self.type = "Sigmoid"

    def forward(self, inputs):
        self.input = inputs
        self.output = 1 / (1 + np.exp(-1 * inputs))

    def backwards(self, dvalues):
        self.dinputs = dvalues * self.output * (1 - self.output)

class Loss():
    def calculate(self, output, y):
        sampleLoss = self.forward(output, y)
        dataLoss = np.mean(sampleLoss)
        return dataLoss
    
    def regularisationLoss(self, layer):
        reguLoss = 0

        if layer.weightRegL1 > 0:
            reguLoss += layer.weightRegL1 * np.sum(np.abs(layer.weights))
    
        if layer.weightRegL2 > 0:
            reguLoss += layer.weightRegL2 * np.sum(layer.weights * layer.weights)

        if layer.biasRegL1 > 0:
            reguLoss += layer.biasRegL1 * np.sum(np.abs(layer.biases))

        if layer.biasRegL2 > 0:
            reguLoss += layer.biasRegL2 * np.sum(layer.biases * layer.biases)

        return reguLoss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, yPred, yTrue):
        sampleSize = len(yPred)
        yPredClipped = np.clip(yPred, 1e-7, 1-(1e-7)) # clips to avoid divide by 0 errors and to avoid dragging the mean to a value
        # only if categorical labels
        if len(yTrue.shape) == 1:
            correct_confidences = yPredClipped[range(sampleSize), yTrue]

        # Mask values - only for one-hot encoded labels
        elif len(yTrue.shape) == 2:
            correct_confidences = np.sum(yPredClipped*yTrue, axis=1)
            
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, yTrue):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(yTrue.shape) == 1:
            yTrue = np.eye(labels)[yTrue]

        self.dinputs = -yTrue / dvalues
        self.dinputs = self.dinputs / samples

class Optimiser_SGD():
    #initialises the parameters
    def __init__(self, learningRate=1.0, decay=0., momentum=0.):
        self.learningRate = learningRate
        self.currentLearningRate = learningRate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    #updating hyperparameters before updating parameters
    def pre_update(self):
        if self.decay:
            self.currentLearningRate = self.learningRate * (1. / (1. + self.decay * self.iterations))

    #updating parameters
    def update_parameters(self, layer):
        if self.momentum:
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weightUpdates = self.momentum * layer.weight_momentums - self.currentLearningRate * layer.dweights
            layer.weight_momentums = weightUpdates

            biasUpdates = self.momentum * layer.bias_momentums - self.currentLearningRate * layer.dbiases
            layer.bias_momentums = biasUpdates

            layer.weights += weightUpdates
            layer.biases += biasUpdates
        else:
            layer.weights += -layer.dweights * self.learningRate
            layer.biases += -layer.dbiases * self.learningRate

    #increment iterations to facilitate updating hyperparameters
    def post_update(self):
        self.iterations += 1


class Optimiser_Adam():
    def __init__(self, learningRate=0.001, decay=0., epsilon=1e-7, beta1=0.9, beta2=0.999):
        self.learningRate = learningRate
        self.currentLearningRate = learningRate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def pre_update(self):
        if self.decay:
            self.currentLearningRate = self.learningRate * (1. / (1. + self.decay * self.iterations))
    
    def update_parameters(self, layer):
        if not hasattr(layer, "weightCache"):
            layer.weightCache = np.zeros_like(layer.weights)
            layer.weightMomentums = np.zeros_like(layer.weights)
            layer.biasCache = np.zeros_like(layer.biases)
            layer.biasMomentums = np.zeros_like(layer.biases)

        layer.weightMomentums = self.beta1 * layer.weightMomentums + (1-self.beta1) * layer.dweights
        layer.biasMomentums = self.beta1 * layer.biasMomentums + (1-self.beta1) * layer.dbiases

        weightMomentumsCorrected = layer.weightMomentums / (1-self.beta1 ** (self.iterations+1))
        biasMomentumsCorrected = layer.biasMomentums / (1-self.beta1 ** (self.iterations+1))

        layer.weightCache = self.beta2 * layer.weightCache + (1-self.beta2) * layer.dweights ** 2
        layer.biasCache = self.beta2 * layer.biasCache + (1-self.beta2) * layer.dbiases ** 2

        weightCacheCorrected = layer.weightCache / (1-self.beta2 ** (self.iterations + 1))
        biasCacheCorrected = layer.biasCache / (1-self.beta2 ** (self.iterations+1))

        layer.weights += -self.currentLearningRate * weightMomentumsCorrected / (np.sqrt(weightCacheCorrected) + self.epsilon)
        layer.biases += -self.currentLearningRate * biasMomentumsCorrected / (np.sqrt(biasCacheCorrected) + self.epsilon)

    def post_update(self):
        self.iterations += 1

#From here on down, is my code entirely
class Model():
    #initialising model variables
    def __init__(self):
        self.layers = []
        self.activations = []
        self.lossFunc = None
        self.optimiser = None
        self.lossFuncType = ""

    def addLayer(self, inputNo, neuronNo, loc=-1, layerType="dense", activation="ReLU", weightRegL1=0, weightRegL2=0, biasRegL1=0, biasRegL2=0, rate=0, weights=None, biases=None):
        if weights is None: #creating a new layer depending on specified type (dense layer by default)
            if layerType == "dense":
                newLayer = Layer_Dense(inputNo, neuronNo, weightRegL1, weightRegL2, biasRegL1, biasRegL2)
            elif layerType == "dropout":
                newLayer = Layer_Dropout(rate,inputNo)
            else:
                raise ValueError("Unimplemented type of layer requested")

            #if the previous layer's output size isnt the same as the provided number of inputs, there is a mistake - only supports fully connected models
            if len(self.layers) > 0:
                if self.layers[-1].neuronNo != inputNo:
                    raise ValueError(f"Layer size mismatch: {self.layers[-1].neuronNo} to {inputNo}")

            #the algorithms used for optimisation only add to the end, but the user has access to this functionality to add a layer wherever they like
            if loc == -1:
                self.layers.append(newLayer)
            else:
                self.layers.insert(loc, newLayer)
                if self.layers[-1].neuronNo != inputNo:
                    raise ValueError(f"Layer size mismatch: {self.layers[-1].neuronNo} to {inputNo}")

            if layerType != "dropout": #if its a dropout layer, there is no activation function
                if activation == "ReLU":
                    newActivation = Activation_ReLU()
                elif activation == "Softmax":
                    newActivation = Activation_Softmax()
                elif activation == "Sigmoid":
                    newActivation = Activation_Sigmoid()
                else:
                    raise ValueError("Unimplemented Activation function needed")
                
                self.activations.append(newActivation)
            else:
                self.activations.append(None)

        else: #adding a layer that is being loaded - only dense layers possible for now
            newLayer = Layer_Dense(inputNo, neuronNo)
            newLayer.weights = weights
            newLayer.biases = biases
            self.layers.append(newLayer)
            if activation == "ReLU":
                newActivation = Activation_ReLU()
            elif activation == "Softmax":
                newActivation = Activation_Softmax()
            elif activation == "Sigmoid":
                newActivation = Activation_Sigmoid()
            else:
                raise ValueError("Unimplemented Activation function needed")
            self.activations.append(newActivation)

    def setLossFunc(self, func="Categorical_Cross_Entropy"):
        if func == "Categorical_Cross_Entropy":
            self.lossFunc = Loss_CategoricalCrossEntropy()
            self.lossFuncType = func
        else:
            raise ValueError("Unimplemented Loss function requested")

    def setOptimiser(self, optimiser="Adam", learningRate=None, momentum=None, decay=None, epsilon=None, beta1=None, beta2=None):
        #sets optimisers using given hyperparameters. If a certain hyperparameters is not given, the default values in the class implementing that optimiser is used
        if optimiser == "SGD":
            kwargs = {}
            if learningRate != None:
                kwargs["learningRate"] = learningRate
            if momentum != None:
                kwargs["momentum"] = momentum
            if decay != None:
                kwargs["decay"] = decay
            self.optimiser = Optimiser_SGD(**kwargs)

        elif optimiser == "Adam":
            kwargs = {}
            if learningRate != None:
                kwargs["learningRate"] = learningRate
            if decay != None:
                kwargs["decay"] = decay
            if epsilon != None:
                kwargs["epsilon"] = epsilon
            if beta1 != None:
                kwargs["beta1"] = beta1
            if beta2 != None:
                kwargs["beta2"] = beta2
            self.optimiser = Optimiser_Adam(**kwargs)

    def forwardPass(self, inputBatch, trueBatch):
        #implements one forward pass on a batch of data
        if self.lossFunc == None:
            raise AttributeError("No loss function set")
        
        self.layers[0].forward(inputBatch)
        self.activations[0].forward(self.layers[0].output)
        self.reguLoss = 0
        for i in range(1,len(self.layers)):
            if self.activations[i-1] != None:
                self.layers[i].forward(self.activations[i-1].output)
            else:
                # if the previous layer was a dropout layer, there will be no activation function, so we skip over this
                self.layers[i].forward(self.layers[i-1].output) 

            if self.activations[i] != None:
                #If this is a dropout layer, we skip the activation function in the pass and we also add its regularisation loss to the count
                self.activations[i].forward(self.layers[i].output)
                self.reguLoss += self.lossFunc.regularisationLoss(self.layers[i])

        #calculate losses and accuracy
        self.dataloss = self.lossFunc.calculate(self.activations[-1].output, trueBatch)
        self.totalLoss = self.dataloss + self.reguLoss
        predictions = np.argmax(self.activations[-1].output, axis=1)
        self.accuracy = np.mean(predictions == trueBatch)
        return self.totalLoss, self.accuracy
    
    def backwardPass(self, trueBatch):
        #backward pass and calculate gradients
        self.lossFunc.backward(self.activations[-1].output, trueBatch)
        self.activations[-1].backward(self.lossFunc.dinputs)
        self.layers[-1].backward(self.activations[-1].dinputs)
        for i in range(len(self.layers)-2, -1, -1):
            if self.activations[i] != None:
                #if this is a dropout layer, we skip activation function
                self.activations[i].backward(self.layers[i+1].dinputs)
                self.layers[i].backward(self.activations[i].dinputs)
            else:
                self.layers[i].backward(self.layers[i+1].dinputs)

        #update parameters
        self.optimiser.pre_update()
        for layer in self.layers:
            if hasattr(layer, "weights"): #if its a dropout layer, it shouldnt be optimised
                self.optimiser.update_parameters(layer)
        self.optimiser.post_update() 

    def cycle(self, x, y): 
        #essentially one epoch of training/validation/testing
        #assuming layers exist, loss func is set and optimiser is set
        if self.layers == [] or self.activations == [] or self.lossFunc == None or self.optimiser == None:
            raise ValueError("Conditions not met to be able to run training.\nCheck if layers, activations, loss func and optimiser is added")
        
        #shuffle since this function will be running multiple times in a loop
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]
        
        x = np.split(x, 1000)
        y = np.split(y, 1000)
        losses = []
        accs = []

        #run backward and forward pass once for each batch
        for i in range(len(x)):
            lossTmp, accTmp = self.forwardPass(x[i], y[i])
            losses.append(lossTmp)
            accs.append(accTmp)
            self.backwardPass(y[i])
        loss = sum(losses) / len(losses)
        acc = sum(accs) / len(accs)
        return loss, acc

    def sizeOptimiser(self, inputNo, outputNo, x, y, xVal, yVal, hiddenActivation="ReLU", outputActivation="Softmax"):
        #first create the initial input, output and singular hidden layer
        #reinitialise the layers and activations in case they were already populated
        print("Optimising size")
        self.layers = []
        self.activations = []
        self.addLayer(inputNo, 64, activation=hiddenActivation)
        self.addLayer(64, outputNo, activation=outputActivation)
        lastAction = None

        #print shape
        print(inputNo, end=", ")
        for layer in self.layers:
            print(layer.neuronNo, end=", ")
        print()

        previousValLoss = 5
        previousLoss = 5
        willBreak = False
        while True:
            #calculate loss and validation loss
            for epoch in range(10):
                loss, acc = self.cycle(x,y)
                print(f"Epoch: {epoch}, loss: {loss}, acc: {acc}")

            deltaLoss = previousLoss - loss
            valLoss, valAcc = self.forwardPass(xVal, yVal)
            print(f"Validation Loss: {valLoss}, Validation Accuracy: {valAcc}")

            if willBreak:
                print("Size optimisation complete")
                break

            deltaValLoss = previousValLoss - valLoss

            if deltaValLoss < 0.1 and lastAction == "addDropOut":
                #even a dropout layer doesnt help anymore, optimisation complete
                #if we are here, the new addition to the size either made things worse or so marginably better that we have identified it negligible
                #Therefore, we should undo our last action - remove the last dropout layer and retrain
                self.layers.pop(len(self.layers)-2)
                self.activations.pop(len(self.activations)-2)
                willBreak = True

            elif (deltaValLoss < 0.1 and loss < valLoss):
                #instead of optimisation complete, this is overfitting - add a dropout layer
                self.layers.pop()
                self.activations.pop()
                prevSize = self.layers[-1].neuronNo
                self.addLayer(prevSize, prevSize, layerType="dropout", rate=0.45)
                self.addLayer(prevSize, outputNo, activation="Softmax")
                lastAction = "addDropOut"

            elif deltaLoss < 0.1:
                #adding new layer could help
                self.layers.pop() # remove the last layer first to avoid neuron number mismatch
                self.activations.pop()
                self.addLayer(self.layers[-1].neuronNo, 64, activation=hiddenActivation, weightRegL2=5e-4, biasRegL2=5e-4)
                self.addLayer(self.layers[-1].neuronNo, outputNo, activation=outputActivation)
                lastAction = "addLayer"

            else:
                #adding size could help - we only ever modify the penultimate layer
                self.layers.pop()
                self.activations.pop()
                prevSize = self.layers[-1].neuronNo
                self.layers.pop() #remove the last 2, since we arent trying to add a layer, just increase an existing ones size
                self.activations.pop()
                newInputSize = self.layers[-1].neuronNo if self.layers else inputNo
                self.addLayer(newInputSize, prevSize*2, activation=hiddenActivation, weightRegL2=5e-4, biasRegL2=5e-4)
                self.addLayer(self.layers[-1].neuronNo, outputNo, activation=outputActivation)
                lastAction = "widen"
                
            #print shape
            print(inputNo, end=", ")
            for layer in self.layers:
                print(layer.neuronNo, end=", ")
            print()

            previousLoss = loss
            previousValLoss = valLoss
        self.sizeOptimised = True
        #print shape
        print(inputNo, end=", ")
        for layer in self.layers:
            print(layer.neuronNo, end=", ")
        print()

    def hyperParamOptimiser(self, x, y, xVal, yVal, paramDict, optimiser): #to be ran AFTER size optimiser

        #reinitialiise the optimiser in case the user input new starting values
        self.setOptimiser(optimiser=optimiser, **paramDict)
        previousValLoss = 5
        
        for param, value in paramDict.items():
            direction = [] #keeps track of whether we made the values bigger or smaller so we can make the correct decisions on how to modify it next
            sens = 1 #sensitivity of the adjustment - the higher this value, the smaller the change to the hyperparameter
            while sens <= 3: #picked 3 arbitrarily due to time and hardware limitations
                for i in range(3):
                    loss, acc = self.cycle(x,y)

                valLoss, valAcc = self.forwardPass(xVal, yVal)
                print(f"Validation Loss: {valLoss}, Validation Accuracy: {valAcc}")
                deltaValLoss = previousValLoss - valLoss

                if len(direction) == 0: #first run - go smaller (I decided to go smaller because this is usually the right direction - we handle if this was incorrect later)
                    newDir = -1
                    paramDict[param] = self.updateHyperParam(param, value, newDir, sens)       
                else:
                    if abs(deltaValLoss) < 0.1:
                        #at this point its good enough
                        print(f"{param} optimised")
                        break
                    elif valLoss < previousValLoss:
                        #then keep going in the same direction
                        newDir = direction[-1]
                        paramDict[param] = self.updateHyperParam(param, value, newDir, sens)
                    else: 
                        newDir = 0 if direction[-1] == 1 else 1
                        #change direction - but use smaller widths - unless its the second turn. It is possible down was the wrong direction
                        if len(direction) == 1:
                            paramDict[param] = self.updateHyperParam(param, value, newDir, 0.5)
                        else:
                            sens += 1
                            paramDict[param] = self.updateHyperParam(param, value, newDir, sens)
                direction.append(newDir)
                previousValLoss = valLoss

    #helper function to hyperparameter optimiser - handles actually changing the values of the hyperparameters
    def updateHyperParam(self, parameter, value, direction, sens):
        if parameter == "learningRate" or parameter == "decay":
            value = value * np.power(10, 1/sens) if direction == 1 else value / np.power(10, 1/sens)
        else:
            mutable = 1 - value
            mutable = mutable * np.power(10, 1/sens) if direction == 1 else mutable / np.power(10, 1/sens)
            value = 1-mutable
        return value


    def saveModel(self):
        #Uses pickle to save the model
        parameters = []
        for i in range(len(self.layers)):
            if self.activations[i] != None: #if its not a dropout layer
                parameters.append((self.layers[i].biases, self.layers[i].weights))

        with open("NNParameters.pkl", "wb") as file:
            pkl.dump(parameters, file)

        with open("NNMetaData.pkl", "wb") as file:
            layerList = []
            for i in range(len(self.layers)):
                if self.activations[i] != None:
                    layerList.append((self.layers[i].type, self.activations[i].type))
                else:
                    layerList.append((self.layers[i].type, None))
            layerList.append((None, self.lossFuncType))

            hyperParams = {}
            hyperParams["learningRate"] = self.optimiser.learningRate
            hyperParams["decay"] = self.optimiser.decay
            if hasattr(self.optimiser, "momentum"):
                hyperParams["momentum"] = self.optimiser.momentum
            if hasattr(self.optimiser, "epsilon"):
                hyperParams["epsilon"] = self.optimiser.epsilon
            if hasattr(self.optimiser, "beta1"):
                hyperParams["beta1"] = self.optimiser.beta1
            if hasattr(self.optimiser, "beta2"):
                hyperParams["beta2"] = self.optimiser.beta2
            
            pkl.dump((layerList, hyperParams), file)

    def loadModel(self, paramFile, metaFile):
        #Uses pickle to load the model
        with open(metaFile, "rb") as file: 
            layerList, hyperParams = pkl.load(file)
        with open(paramFile, "rb") as file:
            parameters = pkl.load(file)

        i = 0 #overall count
        j = 0 #dense layer count
        for (layerType, activationType) in layerList:
            if i < len(layerList) - 1:
                if activationType == None: #dropout layer
                    self.addLayer(self.layers[i-1].neuronNo, self.layers[i-1].neuronNo, layerType="dropout")
                else:
                    weights = parameters[j][1]
                    biases = parameters[j][0]
                    self.addLayer(len(weights), len(biases), layerType=layerType, activation=activationType, weights=weights, biases=biases, weightRegL2=5e-4, biasRegL2=5e-4)
                    j += 1
                i += 1
        self.setLossFunc(layerList[-1][1])
        if "momentum" in hyperParams: #can be expanded if more are added
            self.setOptimiser(optimiser="SGD", **hyperParams)
        else:
            self.setOptimiser(optimiser="Adam", **hyperParams)            
        