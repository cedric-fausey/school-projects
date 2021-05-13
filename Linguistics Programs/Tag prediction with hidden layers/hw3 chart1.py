# does it without normalization layers

import csv
import sys
import operator
import torch
from collections import Counter

import numpy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

trainFile = sys.argv[1]
trainReader = csv.reader(open(trainFile, 'r', encoding="utf8"), delimiter='\t')
trainReader2 = csv.reader(open(trainFile, 'r', encoding="utf8"), delimiter='\t')
devFile = sys.argv[2]
devReader = csv.reader(open(devFile, 'r', encoding="utf8"), delimiter='\t')

chartWriter = csv.writer(open("hw3_hiddenLayers.txt", 'w', encoding="utf8", newline=''), delimiter='\t')
labelsWriter = open("hw3_labels.txt", 'w', encoding="utf8")

batchSize = int(input("Enter batch size: "))
#learningRate = float(input("Enter learning rate: "))
passCount = int(input("Enter pass count: "))
# new to homework 3
hiddenSize = 128
hiddenCount = int(input("Enter number of hidden layers: "))# 1 to 4 I believe

# in this program, features refers to the 1, 2, and 3 letter prefixes and suffixes
featureFrequency = Counter() # used to determine 300 most common
featureIds = {} # only top 300 in this one
featureData = []
tempRowF = []
featureCount = 0

tagIds = {}
tagData = []
tempRowT = []
tagCount = 0

class MyClassifier(torch.nn.Module):
    def __init__(self, featureCount, tagCount, hiddenSize, hiddenCount): # currently one hidden layer, TODO: do it with more layers using an if statement
        super(MyClassifier, self).__init__()
        self.lin = torch.nn.Linear(featureCount, hiddenSize)
        if hiddenCount > 1:
            self.lin2 = torch.nn.Linear(hiddenSize, hiddenSize)
        if hiddenCount > 2:
            self.lin3 = torch.nn.Linear(hiddenSize, hiddenSize)
        if hiddenCount > 3:
            self.lin4 = torch.nn.Linear(hiddenSize, hiddenSize)
        self.linFinal = torch.nn.Linear(hiddenSize, tagCount)
    # modified for chart usage, so that there's a function outputting the last hidden layer as well as a function that forward passes
    def lastHidden(self, batchFeatures, hiddenCount):
        hidden = torch.sigmoid(self.lin(batchFeatures))
        #print("hidden:", hidden)
        if hiddenCount > 1:
            hidden2 = torch.sigmoid(self.lin2(hidden))
            #print("hidden2:", hidden2)
        if hiddenCount > 2:
            hidden3 = torch.sigmoid(self.lin3(hidden2))
            #print("hidden3:", hidden3)
        if hiddenCount > 3:
            hidden4 = torch.sigmoid(self.lin4(hidden3))
            #print("hidden4:", hidden4)
        
        if hiddenCount == 1:
            return hidden
        elif hiddenCount == 2:
            return hidden2
        elif hiddenCount == 3:
            return hidden3
        elif hiddenCount == 4:
            return hidden4
        else:
            return "n/a"
    def forward(self, batchFeatures, hiddenCount): # also should depend on the number of layers
        lastHiddenLayer = self.lastHidden(batchFeatures, hiddenCount)
        if hiddenCount > 0 and hiddenCount <= 4:
            return torch.sigmoid(self.linFinal(lastHiddenLayer))
        else:
            return "n/a"

def getFeatures(word):
    features = []
    for i in range(6):
        features.append("n/a")
    if len(word) == 1:
        features[0] = "pref-" + word
        features[5] = "suff-" + word
    elif len(word) == 2:
        features[0] = "pref-" + word[:1]
        features[1] = "pref-" + word
        features[4] = "suff-" + word
        features[5] = "suff-" + word[-1:]
    elif len(word) >= 3:
        features[0] = "pref-" + word[:1]
        features[1] = "pref-" + word[:2]
        features[2] = "pref-" + word[:3]
        features[3] = "suff-" + word[-3:]
        features[4] = "suff-" + word[-2:]
        features[5] = "suff-" + word[-1:]
    return features

# first goes through the training file to get the top 300 most common features

file1percent = 2457 # manually changed depending on file read - English is 2457

lineNum = 0
for line in trainReader:
    if lineNum % file1percent == 0:
        print("Processing file (part 1):", lineNum/file1percent, "%")
    lineNum += 1
    if len(line) == 10 and line[5] != "_":
        word = line[1]
        tags = line[5].split(";")
        features = getFeatures(word)
        for feature in features:
            if len(feature) > 3:
                featureFrequency.update({feature:1})
                # print(feature, featureFrequency[feature])
        for tag in tags:
            if tag not in tagIds:
                tagIds[tag] = tagCount
                tagCount += 1

#for feature in featureFrequency:
#    print(feature, "-", featureFrequency[feature], "occurrences")

featuresTop300 = featureFrequency.most_common(300)
rank = 0
#print("Most common features:")
for feature, frequency in featuresTop300:
    featureIds[feature] = rank
    rank += 1
    #print(rank, "-", feature)
    #if feature in featureIds:
        #print(feature, "is in featureIds")

# then goes through the training file again and properly assembles the matrices for features and tags

lineNum = 0
exampleNum = 0
featureTensors = []
tagTensors = []

for line in trainReader2:
    if lineNum % file1percent == 0:
        print("Processing file (part 2):", lineNum/file1percent, "%")
    lineNum += 1
    if len(line) == 10 and line[5] != "_":
        exampleNum += 1
        # clear tempRowF, tempRowT, and features (F and T stand for features and tags)
        tempRowF = [0] * len(featureIds)
        tempRowT = [0] * tagCount
        
        # add feature data to temprowF, then add tempRowF to matrix
        word = line[1]
        #print("word:", word)
        features = getFeatures(word)
        for feature in features:
            #print(feature)
            if feature in featureIds:
                tempRowF[featureIds[feature]] = 1
                #print("feature", feature, "added to matrix at column", featureIds[feature])
            #elif feature not in featureIds:
                #print("feature", feature, "is not in the top 300")
        featureData.append(tempRowF)
        
        # add tag data to tempRowT, then add tempRowT to matrix
        tags = line[5].split(";")
        for tag in tags:
            if tag in tagIds:
                tempRowT[tagIds[tag]] = 1
            #print("tag", tag, "added to matrix at column", tagIds[tag])
        tagData.append(tempRowT)
        
        if exampleNum % batchSize == 0:
            featureTensors.append(torch.FloatTensor(featureData))
            tagTensors.append(torch.FloatTensor(tagData))
            featureData = []
            tagData = [] # make sure this works

model = MyClassifier(len(featureIds), len(tagIds), hiddenSize, hiddenCount) # new here
optimizer = torch.optim.Adam(model.parameters())
# optimizer = torch.optim.SGD(model.parameters(), learningRate)
criterion = torch.nn.BCELoss()
every100 = 0

for i in range(passCount): # 100 is for testing
    for j in range(len(featureTensors)):    
        every100 += 1
        model.zero_grad()
        tagPredictions = model.forward(featureTensors[j], hiddenCount) # results of forward pass
        loss = criterion(tagPredictions, tagTensors[j])
        # loss = criterion(torch.sigmoid(tagPredictions), tagTensors[j]) # decreases slightly each iteration
        loss.backward()
        optimizer.step()
        #if j == 1 or j == 2:
        if every100 % 100 == 0:
            print("optimization loop run through", i+1, "times on batch", j)
            #print("model:", model)
            #print("feature tensor:")
            #print(featureTensors[j])
            #print("tag tensor:")
            #print(tagTensors[j])
            print("tagPredictions:")
            print(tagPredictions)
            print("loss:", loss)
            print("optimizer:", optimizer)


tokenCount = 0 # so that only the first 200 lines are read


with torch.no_grad():
    while tokenCount < 200:
        line = next(devReader)
        if len(line) == 10 and line[5] != "_":
            tokenCount += 1
            word = line[1]
            truePos = line[5] + ";"
            guessedPos = ""
            features = getFeatures(word)
            featureOccurrences = [0] * len(featureIds)
            trueTags = truePos.split(';')
            tagOccurrences = [0] * len(tagIds)
            
            for feature in featureIds:
                if feature in features:
                    featureOccurrences[featureIds[feature]] = 1
                    print("feature occurrences updated")
            devFeatureTensor = torch.FloatTensor(featureOccurrences)
            print("devFeatureTensor:", devFeatureTensor)
            
            for tag in tagIds:
                if tag in trueTags:
                    tagOccurrences[tagIds[tag]] = 1
                    print("tag occurrences updated")
            devTagTensor = torch.FloatTensor(tagOccurrences)
            print("devTagTensor:", devTagTensor)
            
            lastHiddenLayer = model.lastHidden(devFeatureTensor, hiddenCount)
            print("lastHiddenLayer:", lastHiddenLayer)
            tagPredictions = model.forward(devFeatureTensor, hiddenCount)
            print("tagPredictions:", tagPredictions)
            loss = criterion(tagPredictions, devTagTensor)
            
            # writes the data to the files
            chartWriter.writerow(lastHiddenLayer.numpy())
            labelsWriter.write(word + '\n')