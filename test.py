from sklearn.ensemble import RandomForestClassifier
import numpy as np
from math import log

domainlist = []
inputlist=[]
class Domain:
	def __init__(self,_name,_label):
		self.name = _name
		self.label = _label


	def returnData(self):
		return [self.name]

	def returnLabel(self):
		if self.label == "dga":
			return 0
		else:
			return 1

def cal_entropy(dataSet):
    shannonEnt = 0.0
    for key in dataSet:
        shannonEnt += ord(key) * log(ord(key), 2) 
    return shannonEnt


	
def initData(filename):
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith("#") or line =="":
				continue
			tokens = line.split(",")
			name = cal_entropy(tokens[0])
			label = tokens[1]
			domainlist.append(Domain(name,label))
			
def readData(filename):
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith("#") or line =="":
				continue
			tokens = line.split(",")
			name = tokens[0]
			inputlist.append(name)
			

def main():
	initData("train.txt")
	readData("test.txt")
	featureMatrix = []
	labelList = []
	for item in domainlist:
		featureMatrix.append(item.returnData())
		labelList.append(item.returnLabel())
	clf = RandomForestClassifier(random_state=1)
	clf.fit(featureMatrix,labelList)
	with open("result.txt","w") as f:
		for i in inputlist:
			k=cal_entropy(i)
			if(clf.predict([[k]])[0]==0):
				f.write(i)
				f.write(",dga")
				f.write("\n")
			else:
				f.write(i)
				f.write(",notdga")
				f.write("\n")
			

if __name__ == '__main__':
	main()

