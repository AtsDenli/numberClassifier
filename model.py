import cupy as np
import sys

#Code is almost identical (there are some small differences since I didn't copy it outright) to the book
#Neural Networks from Scratch in Python by nnfs 
#which is basically the foundation of all of my knowledge about neural networks

class Layer_Dense:
    def __init__(self, inputNo, neuronNo, weightRegL1=0, weightRegL2=0, biasRegL1=0, biasRegL2=0):
        self.inputNo = inputNo
        self.neuronNo = neuronNo
        self.weights = 0.01 * np.random.randn(inputNo, neuronNo)
        #the reason that there is an extra bracket here to specify the shape but not in the rpevious line is that zeros() takes other values in as a parameters. If you did it the other way, 1 would be the shape and neuronNo would be the dataType
        self.biases = np.zeros((1, neuronNo))
        self.weightRegL1 = weightRegL1
        self.weightRegL2 = weightRegL2
        self.biasRegL1 = biasRegL1
        self.biasRegL2 = biasRegL2 

    def forward(self, inputs):
        self.inputs = inputs
        #the reason we dont just do * here instead of np.dot is because we need to be able to apply this to batches
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        #dvalues are the gradients w.r.t inputs from the next layer
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        #now we found the gradients of the weights and biases and the inputs that will be passed to the previous layer

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
    def __init__(self, rate):
        #the input is the rate of dropout, but we deal with the rate of success, so we invert it
        self.rate = 1 - rate
    
    def forward(self, inputs):
        self.inputs = inputs
        #the mask of the things we will keep. whats randomly assigned as a 0 here will deactivate the related neuron
        self.binaryMask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binaryMask

    def backward(self, dvalues):
        #if its 0, the derivativve isdd also 0. else its 1, since the dropout layer is either an active heuron or not, so the values it passes back is 1,
        self.dinputs = dvalues * self.binaryMask

class Activation_ReLU():
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        #self.inputs and not dinputs here since we want to know how the inputs to the 

class Activation_Softmax():
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
        # Probabilities for target values -
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
        #sparse values is when we just have 1 vector telling us the index of each correct result. We want to turn it into a one hot vector
        if len(yTrue.shape) == 1:
            yTrue = np.eye(labels)[yTrue]

        self.dinputs = -yTrue / dvalues
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossEntropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, yTrue):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, yTrue)
    
    def backward(self, dvalues, yTrue):
        samples = len(dvalues)
        #if it is a one hot vector, turn it into a sparse vector
        if len(yTrue.shape) == 2:
            yTrue = np.argmax(yTrue, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), yTrue] -= 1
        self.dinputs = self.dinputs / samples

class Optimiser_SGD():
    def __init__(self, learningRate=1.0, decay=0., momentum=0.):
        self.learningRate = learningRate
        self.currentLearningRate = learningRate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update(self):
        if self.decay:
            self.currentLearningRate = self.learningRate * (1. / (1. + self.decay * self.iterations))
    
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
    def __init__(self):
        self.layers = []
        self.activations = []
        self.lossFunc = None
        self.optimiser = None

    def addLayer(self, inputNo, neuronNo, loc=-1, layerType="dense", activation="ReLU", weightRegL1=0, weightRegL2=0, biasRegL1=0, biasRegL2=0, rate=0):
        if layerType == "dense":
            newLayer = Layer_Dense(inputNo, neuronNo, weightRegL1, weightRegL2, biasRegL1, biasRegL2)
        elif layerType == "dropout":
            newLayer = Layer_Dropout(rate)
        else:
            raise ValueError("Unimplemented type of layer requested")
        
        if len(self.layers) > 0:
            if self.layers[-1].neuronNo != inputNo:
                raise ValueError(f"Layer size mismatch: {self.layers[-1].neuronNo} to {inputNo}")
        
        if loc == -1:
            self.layers.append(newLayer)
        else:
            self.layers.insert(loc, newLayer)

        if layerType != "dropout": #if its a dropout layer, there is no activation function
            if activation == "ReLU":
                newActivation = Activation_ReLU()
            elif activation == "Softmax":
                newActivation = Activation_Softmax()
            elif activation == "Sigmoid":
                newActivation = Activation_Sigmoid()
            else:
                raise ValueError("Unimplemented Activation function requested")
            
            self.activations.append(newActivation)
        else:
            self.activations.append(None)

    def setLossFunc(self, func="Categorical_Cross_Entropy"):
        if func == "Categorical_Cross_Entropy":
            self.lossFunc = Loss_CategoricalCrossEntropy()
        else:
            raise ValueError("Unimplemented Loss function requested")
        #if I implement more loss functions, they can be added here

    def setOptimiser(self, optimiser="Adam", learningRate=None, momentum=None, decay=None, epsilon=None, beta1=None, beta2=None):
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

        self.dataloss = self.lossFunc.calculate(self.activations[-1].output, trueBatch)
        self.totalLoss = self.dataloss + self.reguLoss
        predictions = np.argmax(self.activations[-1].output, axis=1)
        self.accuracy = np.mean(predictions == trueBatch)
        return self.totalLoss, self.accuracy
    
    def backwardPass(self, trueBatch):
        self.lossFunc.backward(self.activations[-1].output, trueBatch)
        self.activations[-1].backward(self.lossFunc.dinputs)
        self.layers[-1].backward(self.activations[-1].dinputs)
        for i in range(len(self.layers)-2, -1, -1):
            if self.activations[i] != None:
                #if this is a dropout layer, we skip activation function
                self.activations[i].backward(self.layers[i+1].dinputs)

            if self.activations[i+1] != None:
                # if the next layer is dropout, we will use the layer instead of activation
                self.layers[i].backward(self.activations[i].dinputs)
            else:
                self.layers[i].backward(self.layers[i+1].dinputs)

        self.optimiser.pre_update()
        for layer in self.layers:
            self.optimiser.update_parameters(layer)
        self.optimiser.post_update() 

    def cycle(self, x, y): #essentially one epoch of training/validation/testing
        #assuming layers exist, loss func is set and optimiser is set
        if self.layers == [] or self.activations == [] or self.lossFunc == None or self.optimiser == None:
            raise ValueError("Conditions not met to be able to run training.\nCheck if layers, activations, loss func and optimiser is added")
        
        #shuffe since this function will be running multiple times in a loop
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]
        #im arbitrarily setting batch size to 2000
        np.split(x, 2000)
        np.split(y, 2000)
        losses = []
        accs = []
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
        self.layers = []
        self.activation = []
        self.addLayer(inputNo, 64, activation=hiddenActivation)
        self.addLayer(64, outputNo, activation=outputActivation)
        depthExhausted = False
        widthExhausted = False
        lastAction = None

        previousValLoss = 5
        previousLoss = 5
        while True:
            for epoch in range(1000):
                loss, acc = self.cycle(x,y)
                print(f"Epoch: {epoch}, loss: {loss}, acc: {acc}")

            deltaLoss = ((previousLoss - loss) / previousLoss) * 100
            valLoss, valAcc = self.cycle(xVal, yVal)
            deltaValLoss = ((previousValLoss - valLoss) / previousValLoss) * 100

            if lastAction == "addLayer" and deltaLoss < 0.01:
                depthExhausted = True   # depth didn't help either -> now genuinely exhausted
            if lastAction == "widen" and deltaLoss < 0.01:
                widthExhausted = True

            if (deltaValLoss < 0.01 and deltaLoss > 0.01) or widthExhausted and depthExhausted:
                #adding more layers is useless, increasing size is useless - optimisation complete
                print("Optimal Size reached")
                break

            elif deltaLoss < 0.01:
                widthExhausted = True
                if not depthExhausted:
                #adding new layer could help
                    self.layers.pop() # remove the last layer first to avoid neuron number mismatch
                    self.addLayer(self.layers[-1].neuronNo, 64, activation=hiddenActivation)
                    self.addLayer(self.layers[-1].neuronNo, outputNo, activation=outputActivation)
                    lastAction = "addLayer"

            else:
                #adding size could help - we only ever modify the penultimate layer
                widthExhausted = False
                self.layers.pop()
                prevSize = self.layers[-1].neuronNo
                self.layers.pop() #remove the last 2, since we arent trying to add a layer, just increase an existing ones size
                newInputSize = self.layers[-1].neuronNo if self.layers else inputNo
                self.addLayer(newInputSize, prevSize*2, activation=hiddenActivation)
                self.addLayer(self.layers[-1].neuronNo, outputNo, activation=outputActivation)
                lastAction = "widen"
                
            previousLoss = loss
            previousValLoss = valLoss