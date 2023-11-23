import argparse
import os
import pickle
import random
import pandas as pd
import numpy as np
import json

def load_data(filename):
    print("read from:", filename)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def sava_data(filename, data):
    print("save to:", filename)
    data.to_pickle(filename)

def processFour(vulpath, goodpath):
    vulfiles = os.listdir(vulpath)
    goodfiles = os.listdir(goodpath)
    allData = [[], [], [], [], [], [], [], []]
    for name in vulfiles:
        data = load_data(vulpath + "/" + name)
        if name.startswith("CVE_raw"):
            pos = 0
        elif name.startswith("CVE"):
            pos = 1
        elif name.startswith("reveal"):
            pos = 2
            # continue
        else:
            pos = 3
        allData[pos].append({
            "codeVector": data[0],
            "sentenceArray": data[1],
            "indegreeArray": data[2],
            "isvul": 1})
    for name in goodfiles:
        data = load_data(goodpath + "/" + name)
        if name.startswith("raw"):
            pos = 4
        elif name.startswith("CVE"):
            pos = 5
        elif name.startswith("reveal"):
            pos = 6
            # continue
        else:
            pos = 7
        allData[pos].append({
            "codeVector": data[0],
            "sentenceArray": data[1],
            "indegreeArray": data[2],
            "isvul": 0})
    for each in allData:
        print(len(each))
    return allData

def generateThreePartSplit(dataList, outputFloder, tr, vr):
    splitTrain1=[]
    splitVal1 = []
    splitTest1 = []
    splitTrain2=[]
    splitVal2 = []
    splitTest2 = []
    for i in range(len(dataList)):
        each = dataList[i]
        if len(each)>0:
            random.shuffle(each)
            splitPos1 = int(len(each) * tr)
            splitPos2 = int(len(each) * tr+vr)
            thisTrain=each[:splitPos1]
            thisVal=each[splitPos1:splitPos2]
            thisTest=each[splitPos2:]
            if i==0 or i==4:
                splitTrain1+=thisTrain
                splitVal1+=thisVal
                splitTest1+=thisTest
            elif i==3 or i==7:
                splitTrain2+=thisTrain
                splitVal2+=thisVal
                splitTest2+=thisTest
    random.shuffle(splitTrain1)
    train1=pd.DataFrame(splitTrain1)
    random.shuffle(splitVal1)
    val1=pd.DataFrame(splitVal1)
    random.shuffle(splitTest1)
    test1=pd.DataFrame(splitTest1)

    sava_data(outputFloder + "/train1.pickle", train1)
    sava_data(outputFloder + "/val1.pickle", val1)
    sava_data(outputFloder + "/test1.pickle", test1)

    random.shuffle(splitTrain2)
    train2=pd.DataFrame(splitTrain2)
    random.shuffle(splitVal2)
    val2=pd.DataFrame(splitVal2)
    random.shuffle(splitTest2)
    test2=pd.DataFrame(splitTest2)

    sava_data(outputFloder + "/train2.pickle", train2)
    sava_data(outputFloder + "/val2.pickle", val2)
    sava_data(outputFloder + "/test2.pickle", test2)

    allDataTrain=splitTrain1+splitTrain2
    del splitTrain1,splitTrain2
    allDataVal=splitVal1+splitVal2
    del splitVal1,splitVal2
    allDataTest=splitTest1+splitTest2
    del splitTest1,splitTest2
    random.shuffle(allDataTrain)
    train = pd.DataFrame(allDataTrain)
    print(train.iloc[0:5, 0:])
    print(len(allDataTrain))
    random.shuffle(allDataVal)
    val = pd.DataFrame(allDataVal)
    print(val.iloc[0:5, 0:])
    print(len(allDataVal))
    random.shuffle(allDataTest)
    test = pd.DataFrame(allDataTest)
    print(test.iloc[0:5, 0:])
    print(len(test))
    sava_data(outputFloder + "/train.pickle", train)
    sava_data(outputFloder + "/val.pickle", val)
    sava_data(outputFloder + "/test.pickle", test)

def parse_options():
    parser = argparse.ArgumentParser(description='Extracting Cpgs.')
    parser.add_argument('-v', '--vul', help='The dir path of dot file', type=str, required=True)
    parser.add_argument('-n', '--noVul', help='The dir path of source code', type=str, required=True)
    parser.add_argument('-o', '--output', help='The dir path of output', type=str, required=True)
    parser.add_argument('-tr', '--trainRate', help='The rate of train data and all dataset', required=True)
    parser.add_argument('-vr', '--valRate', help='The rate of val data and all dataset', required=True)
    args = parser.parse_args()
    return args
def main():
    args = parse_options()
    datas = processFour(args.vul, args.noVul)
    generateThreePartSplit(datas, args.output, args.trainRate, args.valRate)

if __name__ == '__main__':
    main()

# python pklGenNew.py -v /media/zhengboyang/数据/vulcnn-data/data118/outputs/Vul -n /media/zhengboyang/数据/vulcnn-data/data118/outputs/No-Vul -o /media/zhengboyang/数据/vulcnn-data/data118/pkls -tr 0.8 -vr 0.1