import csv
import sys
import operator
import torch
from collections import Counter

trainFile = sys.argv[1]
trainReader = csv.reader(open(trainFile, 'r', encoding="utf8"), delimiter='\t')
trainReader2 = csv.reader(open(trainFile, 'r', encoding="utf8"), delimiter='\t')
devFile = sys.argv[2]
devReader = csv.reader(open(devFile, 'r', encoding="utf8"), delimiter='\t')

batchSize = int(input("Enter batch size: "))
#learningRate = float(input("Enter learning rate: ")) # !!!!!!!! TODO: ask if learning rate is a factor I need to account for in homework 3
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
    def forward(self, batchFeatures, hiddenCount): # also should depend on the number of layers
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
            return torch.sigmoid(self.linFinal(hidden))
            #final = torch.sigmoid(self.linFinal(hidden))
            #print("final:", final)
            #return final1
        elif hiddenCount == 2:
            return torch.sigmoid(self.linFinal(hidden2))
        elif hiddenCount == 3:
            return torch.sigmoid(self.linFinal(hidden3))
        elif hiddenCount == 4:
            return torch.sigmoid(self.linFinal(hidden4))
            #final = torch.sigmoid(self.linFinal(hidden4))
            #print("final:", final)
            #return final
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

# finally reads the dev file and predicts tags based on features

instanceCount = {} # for each of these dictionaries, the key "all" gives the overall count for all tags
accurateInstanceCount = 0
tpCount = {}
fpCount = {}
fnCount = {}
tnCount = {}
accuracy = {}
precision = {}
recall = {}
fScore = {}

for tag in tagIds:
    instanceCount[tag] = 0
    tpCount[tag] = 0
    fpCount[tag] = 0
    fnCount[tag] = 0
    tnCount[tag] = 0
    accuracy[tag] = 0
    precision[tag] = 0
    recall[tag] = 0
    fScore[tag] = 0

instanceCount["all"] = 0
tpCount["all"] = 0
fpCount["all"] = 0
fnCount["all"] = 0
tnCount["all"] = 0
accuracy["all"] = 0
precision["all"] = 0
recall["all"] = 0
fScore["all"] = 0

with torch.no_grad():
    for line in devReader:
        if len(line) == 10 and line[5] != "_":
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
            
            for tag in tagIds:
                if tag in trueTags:
                    tagOccurrences[tagIds[tag]] = 1
                    print("tag occurrences updated")
            devTagTensor = torch.FloatTensor(tagOccurrences)
            
            tagPredictions = model.forward(devFeatureTensor, hiddenCount)
            loss = criterion(tagPredictions, devTagTensor)
            #optimizer.step()
            print("word:", word)
            #print("loss:", loss.item())
            
            for tag in tagIds:
                tagProb = tagPredictions[tagIds[tag]].item()
                if tag in trueTags:
                    print(word, "contains", tag)
                    print(word, "probability of", tag, ":", tagProb) # to test different things, change indent
                if tagProb > 0.5:
                    guessedPos += tag + ";"
                    
            # variables reused from HW1 part 3
            indivTpCount = 0
            indivFpCount = 0
            indivFnCount = 0
            
            instanceCount["all"] += 1
            guessedTags = guessedPos.split(';')
            trueTags = truePos.split(';')\
            
            print("guessed part of speech:", guessedPos)
            print("true part of speech:", truePos)
            
            # same as in HW1 part 3, working both with each tag and as a whole
            guessedTags = guessedPos.split(';')
            trueTags = truePos.split(';')
            for tag in tagIds:
                instanceCount[tag] += 1
                if tag in trueTags:
                    if tag in guessedTags:
                        tpCount[tag] += 1
                        tpCount["all"] += 1
                        indivTpCount += 1
                    elif tag not in guessedTags:
                        fnCount[tag] += 1
                        fnCount["all"] += 1
                        indivFnCount += 1
                if tag not in trueTags:
                    if tag in guessedTags:
                        fpCount[tag] += 1
                        fpCount["all"] += 1
                        indivFpCount += 1
                    else:
                        tnCount[tag] += 1
                        tnCount["all"] += 1
            if indivTpCount > 0 and indivFpCount == 0 and indivFnCount == 0:
                accurateInstanceCount += 1

# per-tag statistics

for tag in tpCount: # includes "all"
    if tpCount[tag] > 0: # used in case of dividing by 0
        precision[tag] = tpCount[tag] / (tpCount[tag] + fpCount[tag])
        recall[tag] = tpCount[tag] / (tpCount[tag] + fnCount[tag])
        fScore[tag] = 2 * tpCount[tag] / (2 * tpCount[tag] + fpCount[tag] + fnCount[tag])
    if tag == "all":
        accuracy[tag] = accurateInstanceCount/instanceCount[tag]
    else:
        accuracy[tag] = (tpCount[tag] + tnCount[tag]) / instanceCount[tag]
        print("Tag:", tag)
        print("   True positive count:   ", tpCount[tag])
        print("   False positive count:  ", fpCount[tag])
        print("   False negative count:  ", fnCount[tag])
        print("   True negative count:   ", tnCount[tag])
        print("   Accuracy of tag:       ", accuracy[tag])
        print("   Precision:             ", precision[tag])
        print("   Recall:                ", recall[tag])
        print("   Micro-averaged f-score:", fScore[tag])

# overall statistics

print("Statistics for entire file:")
print("   Accuracy:              ", accuracy["all"])
print("   Precision:             ", precision["all"])
print("   Recall:                ", recall["all"])
print("   Micro-averaged f-score:", fScore["all"])
