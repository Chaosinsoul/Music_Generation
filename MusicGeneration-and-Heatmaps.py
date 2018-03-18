import random
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
#Parameters
H = 100 #hidden size
layers = 1 # number of lsmt layers
B = 15 # batch size
sequenceLength = 25 #length of sequences trained on
learningRate = 0.01 #learning rate of optimizer
moment = 0.9 #momentum of Standard Gradient Descent
disableTrainLossForSpeed = 0 #speeds up training but doesnt track train loss if set to anything but 0
testSeqLen = 25 #length of sequences tested on
epochs = 1000 #sets the maximum number of epochs in case early stopping fails
epochsPerAccLossTest = 10 #how often validation is tested and recorded
heat = 1 # temperature value used during music file generation

# Build and Training Model Start
# Get Input
# Convert text input to file strings with chars stored in int form
input = open("input.txt", "r")
inputData = []
inputDataString = []
index = 0
smallest = 10000
for line in input:
    if (line == "<start>\n"):
        inputDataString.append("$")
    elif (line == "<end>\n" or line == "<end>"):
        inputDataString[index] = inputDataString[index] + "`"
        index = index + 1
    else:
        inputDataString[index] = inputDataString[index] + line
            
#split file strings to train and validation sets
random.shuffle(inputData)
eightyTwenty = int(index * 0.8)
trainDataString = inputDataString[:eightyTwenty]
testDataString = inputDataString[eightyTwenty:]

#concat strings in each set to 1 large string
combinedTrainString = ""
for i in range(len(trainDataString)):
    combinedTrainString = combinedTrainString + trainDataString[i]
combinedTestString = ""
for i in range(len(testDataString)):
    combinedTestString = combinedTestString + testDataString[i]

#Build converters to convert char to a vocab number and back
chars = sorted(list(set((combinedTrainString + combinedTestString))))
num_vocab = len(chars)
#Create Converters
char_to_idx = dict((c,i) for i,c in enumerate(chars))
idx_to_char = dict((i,c) for i,c in enumerate(chars))

trainSeq = []
for i in range(len(combinedTrainString)):
    trainSeq.append(char_to_idx[combinedTrainString[i]])
testSeq = []
for i in range(len(combinedTestString)):
    testSeq.append(char_to_idx[combinedTestString[i]])
trainSeq = np.array(trainSeq)
testSeq = np.array(testSeq)

#Model

class MusicGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_size, output_dim):
        super(MusicGenerator, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.vocab_size = output_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        
        self.lstm = nn.GRU(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers = num_layers, 
                            batch_first=True)
        self.dropout = nn.Dropout(p=0.0)
        self.scores = nn.Linear(hidden_dim, output_dim)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        
    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.Tensor(self.num_layers,batch_size, self.hidden_dim).zero_()).cuda(),
                autograd.Variable(torch.Tensor(self.num_layers,batch_size, self.hidden_dim).zero_()).cuda())
    
    def forward(self, X, hidden):
        output, hidden = self.lstm(X, hidden[0])

        output = self.dropout(output)
        out_size = self.batch_size * X.size()[1]
        output = output.view(out_size, -1)
    
        output = self.scores(output)
        output = output.view(self.batch_size,self.vocab_size, -1)
        return output, hidden
    
#Functions to create proper input and output tensors based on parameters
def getInput(seqStartPos, trainSeq, batch_size, seq_len, n_vocab):
    tensor = torch.FloatTensor(batch_size, seq_len, n_vocab).zero_()
    curr = seqStartPos
    if (batch_size == 1):
        for i in range(seq_len):
            tensor[0, i, trainSeq[curr]] = 1
            curr = curr + 1
    else:
        for j in range(batch_size):
            for i in range(seq_len):
                tensor[j, i, trainSeq[curr]] = 1
                curr = curr + 1
    data = autograd.Variable(tensor)
    return data

def getOutput(seqStartPos, trainSeq, batch_size, seq_len, n_vocab):
    tensor = torch.LongTensor(batch_size,seq_len).zero_()
    curr = seqStartPos + 1
    if (batch_size == 1):
        for i in range(seq_len):
            tensor[0, i] = int(trainSeq[curr])
    else:
        for j in range(batch_size):
            for i in range(seq_len):
                tensor[j, i] = int(trainSeq[curr])
                curr = curr + 1
    data = autograd.Variable(tensor)
    return data

#create model
model = MusicGenerator(num_vocab,H,layers,B,num_vocab)
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum = moment, nesterov = True)
#optimizer = optim.RMSprop(model.parameters())
#optimizer = optim.Adagrad(model.parameters(), lr=learningRate, lr_decay=0, weight_decay=0)
print(model)
loss_function = nn.CrossEntropyLoss()

#Train
VAL_LOSS = [] 
TRAIN_LOSS = []
VAL_ACC = [] 
TRAIN_ACC = []

#split data for faster testing
trainOutInp = torch.FloatTensor(int(len(trainSeq)/(testSeqLen*B)), B, testSeqLen, num_vocab).zero_()
trainOutTar = torch.LongTensor(int(len(trainSeq)/(testSeqLen*B)), B, testSeqLen).zero_()
for j in range(int(len(trainSeq)/(testSeqLen*B))):
    seqStartPos = j*testSeqLen*B
    trainOutInp[j] = getInput(seqStartPos, trainSeq, B, testSeqLen, num_vocab).data
    trainOutTar[j] = getOutput(seqStartPos, trainSeq, B, testSeqLen, num_vocab).data
testOutInp = torch.FloatTensor(int(len(testSeq)/(testSeqLen*B)), B, testSeqLen, num_vocab).zero_()
testOutTar = torch.LongTensor(int(len(testSeq)/(testSeqLen*B)), B, testSeqLen).zero_()
for j in range(int(len(testSeq)/(testSeqLen*B))):
    seqStartPos = j*testSeqLen*B
    testOutInp[j] = getInput(seqStartPos, testSeq, B, testSeqLen, num_vocab).data
    testOutTar[j] = getOutput(seqStartPos, testSeq, B, testSeqLen, num_vocab).data

for epoch in range(epochs):
    numbatches = int(len(trainSeq)/(sequenceLength*B))
    for j in range(numbatches):
        #one iteration
        hidden = model.init_hidden(B)
        seqStartPos = j*sequenceLength*B
        model.zero_grad()
        inp = autograd.Variable(getInput(seqStartPos, trainSeq, B, sequenceLength, num_vocab).data).cuda()
        tar = autograd.Variable(getOutput(seqStartPos, trainSeq, B, sequenceLength, num_vocab).data).long().cuda()
        tag_scores, hidden = model(inp, hidden)
        tag_scores = tag_scores.view(B*sequenceLength, -1)
        tar = tar.view(B*sequenceLength)
        loss = loss_function(tag_scores, tar)
        loss.backward(retain_graph=True)
        optimizer.step()
    for j in range(layers):
        hidden[j].detach_()
    
    print(epoch+1)
    
    #calc acc & loss, takes a long time dont do too often
    if ((epoch+1) % epochsPerAccLossTest == 0):
        
        print(epoch+1," testing")
        
        if(disableTrainLossForSpeed == 0):
            correct = 0
            total = 0
            trainLoss = 0
            for j in range(int(len(trainSeq)/(testSeqLen*B))):
                hidden = model.init_hidden(B)
                inp = autograd.Variable(trainOutInp[j]).cuda()
                tar = autograd.Variable(trainOutTar[j]).long().cuda()
                tag_scores, hidden = model(inp, hidden)
                tag_scores = tag_scores.view(B*testSeqLen, -1)
                tar = tar.view(B*testSeqLen)
                trainLoss = trainLoss + loss_function(tag_scores, tar).data.cpu().numpy()
                _, preds = torch.max(tag_scores.data, 1)
                for i in range(len(preds)):
                    total = total + 1
                    if (preds[i] == tar.data[i]):
                        correct = correct + 1
            TRAIN_ACC.append(correct/total)
            TRAIN_LOSS.append((trainLoss)/int(len(trainSeq)/(testSeqLen*B)))
        
        correct = 0
        total = 0
        valLoss = 0
        for j in range(int(len(testSeq)/(testSeqLen*B))):
            hidden = model.init_hidden(B)
            inp = autograd.Variable(testOutInp[j]).cuda()
            tar = autograd.Variable(testOutTar[j]).long().cuda()
            tag_scores, hidden = model(inp, hidden)
            tag_scores = tag_scores.view(B*testSeqLen, -1)
            tar = tar.view(B*testSeqLen)
            valLoss = valLoss + loss_function(tag_scores, tar).data.cpu().numpy()
            _, preds = torch.max(tag_scores.data, 1)
            for i in range(len(preds)):
                total = total + 1
                if (preds[i] == tar.data[i]):
                    correct = correct + 1
        VAL_ACC.append(correct/total)
        VAL_LOSS.append((valLoss)/int(len(testSeq)/(testSeqLen*B)))
        if(len(VAL_LOSS) > 2):
            if((VAL_LOSS[len(VAL_LOSS)-1] - VAL_LOSS[len(VAL_LOSS)-2])[0] > 0 and (VAL_LOSS[len(VAL_LOSS)-2] - VAL_LOSS[len(VAL_LOSS)-3])[0] > 0):
                break
                
        print(epoch+1,"end")
        
#Plot Accuracy and loss
import matplotlib.pyplot as plt

xPlot = []
for i in range(len(VAL_LOSS)):
    xPlot.append((i+1)*epochsPerAccLossTest)

## Plot of the accuracy over training for the training, validation and test sets
plt.figure(figsize=(20,10))
if(disableTrainLossForSpeed == 0):
	plt.plot(xPlot, TRAIN_LOSS, label='Train')
plt.plot(xPlot, VAL_LOSS, label='Validation')
plt.xlabel('epoch')
plt.ylabel('Average loss per batch')
plt.legend(shadow=True, prop={'size': 12})
plt.title("Loss for training, validation and testing set")
plt.show()

## Plot of the accuracy over training for the training, validation and test sets
plt.figure(figsize=(20,10))
if(disableTrainLossForSpeed == 0):
	plt.plot(xPlot, TRAIN_ACC, label='Train')
plt.plot(xPlot, VAL_ACC, label='Validation')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend(shadow=True, prop={'size': 12})
plt.title("Acc for training, validation and testing set")
plt.show()
# Build and Training Model End

# Generate Music Start
#Run this to generate a music file

def darkSelect(vals,temp):
    darkProbs = np.exp(vals/temp)
    darkTotal = 0
    for i in range(len(vals[0])):
        darkTotal = darkTotal + darkProbs[0][i][0]
    
    for i in range(len(vals[0])):
        darkProbs[0][i][0] = darkProbs[0][i][0] / darkTotal
    
    pick = random.uniform(0, 1)
    selection = -1
    for i in range(len(vals[0])):
        if(darkProbs[0][i][0] > pick):
            selection = i
            break
        pick = pick - darkProbs[0][i][0]
    return selection

model.batch_size = 1
hidden = autograd.Variable(torch.Tensor(model.num_layers,1, model.hidden_dim).zero_()).cuda()
out = ""
curr = char_to_idx["$"]
heatstring = []
heatvals = []

while (1==1):
    inp = torch.FloatTensor(1, 1, num_vocab).zero_()
    inp[0, 0, curr] = 1
    inp = autograd.Variable(inp)
    inp = inp.cuda()
    
    output, hidden = model.lstm(inp, hidden)
    output = model.dropout(output)
    out_size = model.batch_size * inp.size()[1]
    output = output.view(out_size, -1)
    output = model.scores(output)
    tag_scores = output.view(model.batch_size,model.vocab_size, -1)
    
    curr = darkSelect(tag_scores.data,heat)
    if(idx_to_char[curr] == '`'):
        break
    out = out + idx_to_char[curr]
    heatvals.append((hidden[0].data).cpu().numpy()[0][0])
    heatstring.append(idx_to_char[curr])
print(out)

#extend output so it can be printed as a square matrix
while(1==1):
    if(len(heatvals) % 20 == 0):
        break
    heatvals.append(0)
    heatstring.append(" ")
# Generate Music End

# Print heat maps Start
#Print heat maps
for neuronNum in range(100):
    model.batch_size = 1
    hidden = autograd.Variable(torch.Tensor(model.num_layers,1, model.hidden_dim).zero_()).cuda()
    while(1==1):
        if(heatvals[len(heatvals)-1] == 0):
            heatvals.pop()
            heatstring.pop()
        else:
            break

    heatvals = []
    for i in range(len(heatstring)):
        inp = torch.FloatTensor(1, 1, num_vocab).zero_()
        inp[0, 0, char_to_idx[heatstring[i]]] = 1
        inp = autograd.Variable(inp)
        inp = inp.cuda()

        output, hidden = model.lstm(inp, hidden)
        heatvals.append((hidden[0].data).cpu().numpy()[0][neuronNum])

    while(1==1):
        if(len(heatvals) % 20 == 0):
            break
        heatvals.append(0.0)
        heatstring.append(" ")
    #currently setting manually
    width = (len(heatvals) / 20)
    plt.figure(figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')
    data = np.reshape(heatvals, (int(width), 20))
    heatmap = plt.pcolor(data)
    curr = 0
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            plt.text(x + 0.5, y + 0.5, heatstring[curr], horizontalalignment='center', verticalalignment='center')
            curr = curr + 1

    plt.colorbar(heatmap)
    plt.set_cmap('RdBu')
    plt.show()
    print(neuronNum)
# Print heat maps End