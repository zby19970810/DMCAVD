import argparse
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import keras.backend as K
import random

from gensim.models import Word2Vec


INPUT_SIZE = 500
WORDS_SIZE = 5000
SENTENCE_SIZE = 128
SENTENCE_LENGTH = 256
embedding_dim = 32

random_weights = np.random.normal(size=(WORDS_SIZE, embedding_dim), scale=0.01)

def findKthLargest(nums, k):
    def partition(arr, low, high):
        pivot = arr[low]  # 选取最左边为pivot

        left, right = low, high  # 双指针
        while left < right:

            while left < right and arr[right] >= pivot:  # 找到右边第一个<pivot的元素
                right -= 1
            arr[left] = arr[right]  # 并将其移动到left处

            while left < right and arr[left] <= pivot:  # 找到左边第一个>pivot的元素
                left += 1
            arr[right] = arr[left]  # 并将其移动到right处

        arr[left] = pivot  # pivot放置到中间left=right处
        return left

    def randomPartition(arr, low, high):
        pivot_idx = random.randint(low, high)  # 随机选择pivot
        arr[low], arr[pivot_idx] = arr[pivot_idx], arr[low]  # pivot放置到最左边
        return partition(arr, low, high)  # 调用partition函数

    def topKSplit(arr, low, high, k):
        # mid = partition(arr, low, high)                   # 以mid为分割点【非随机选择pivot】
        mid = randomPartition(arr, low, high)  # 以mid为分割点【随机选择pivot】
        if mid == k - 1:  # 第k小元素的下标为k-1
            return arr[mid]  # 【找到即返回】
        elif mid < k - 1:
            return topKSplit(arr, mid + 1, high, k)  # 递归对mid右侧元素进行排序
        else:
            return topKSplit(arr, low, mid - 1, k)  # 递归对mid左侧元素进行排序

    n = len(nums)
    return topKSplit(nums, 0, n - 1, n - k + 1)  # 第k大元素即为第n-k+1小元素

def findKthSmallest(nums, k):
    def partition(arr, low, high):
        pivot = arr[low]  # 选取最左边为pivot

        left, right = low, high  # 双指针
        while left < right:

            while left < right and arr[right] >= pivot:  # 找到右边第一个<pivot的元素
                right -= 1
            arr[left] = arr[right]  # 并将其移动到left处

            while left < right and arr[left] <= pivot:  # 找到左边第一个>pivot的元素
                left += 1
            arr[right] = arr[left]  # 并将其移动到right处

        arr[left] = pivot  # pivot放置到中间left=right处
        return left

    def randomPartition(arr, low, high):
        pivot_idx = random.randint(low, high)  # 随机选择pivot
        arr[low], arr[pivot_idx] = arr[pivot_idx], arr[low]  # pivot放置到最左边
        return partition(arr, low, high)  # 调用partition函数

    def topKSplit(arr, low, high, k):
        # mid = partition(arr, low, high)                   # 以mid为分割点【非随机选择pivot】
        mid = randomPartition(arr, low, high)  # 以mid为分割点【随机选择pivot】
        if mid == k - 1:  # 第k小元素的下标为k-1
            return arr[mid]  # 【找到即返回】
        elif mid < k - 1:
            return topKSplit(arr, mid + 1, high, k)  # 递归对mid右侧元素进行排序
        else:
            return topKSplit(arr, low, mid - 1, k)  # 递归对mid左侧元素进行排序

    n = len(nums)
    return topKSplit(nums, 0, n - 1, k)  # 这里改成找第k小元素

class addSelf(tf.keras.layers.Layer):
    def __init__(self, sequence_length, **kwargs):
        super(addSelf, self).__init__(**kwargs)
        self.E = np.zeros(shape=(sequence_length, sequence_length))
        for i in range(sequence_length):
            self.E[i][i]=1

    def call(self, inputs):
        return inputs+self.E

class doubleSideAttnWithOri(tf.keras.layers.Layer):
    def __init__(self, input_length, oriHeadNum,inMaskHeadNum,outMaskHeadNum, **kwargs):
        super(doubleSideAttnWithOri, self).__init__(**kwargs)
        self.length = input_length
        self.oriHead = oriHeadNum
        self.inMaskHead = inMaskHeadNum
        self.outMaskHead = outMaskHeadNum
        allHead=inMaskHeadNum+outMaskHeadNum
        if oriHeadNum>0:
            self.keyProduceOri=[]
            for i in range(oriHeadNum):
                self.keyProduceOri.append(tf.keras.layers.Dense(input_length / oriHeadNum, activation='relu'))
            self.queueProduceOri =[]
            for i in range(oriHeadNum):
                self.queueProduceOri.append(tf.keras.layers.Dense(input_length / oriHeadNum, activation='relu'))
        if inMaskHeadNum>0:
            self.valueProduceIn = []
            for i in range(inMaskHeadNum):
                self.valueProduceIn.append(tf.keras.layers.Dense(input_length / allHead, activation='relu'))
            self.queueProduceIn = []
            for i in range(inMaskHeadNum):
                self.queueProduceIn.append(tf.keras.layers.Dense(input_length / allHead, activation='relu'))
            self.keyProduceIn = []
            for i in range(inMaskHeadNum):
                self.keyProduceIn.append(tf.keras.layers.Dense(input_length / allHead, activation='relu'))
        if outMaskHeadNum>0:
            self.valueProduceOut = []
            for i in range(outMaskHeadNum):
                self.valueProduceOut.append(tf.keras.layers.Dense(input_length / allHead, activation='relu'))
            self.queueProduceOut = []
            for i in range(outMaskHeadNum):
                self.queueProduceOut.append(tf.keras.layers.Dense(input_length / allHead, activation='relu'))
            self.keyProduceOut = []
            for i in range(outMaskHeadNum):
                self.keyProduceOut.append(tf.keras.layers.Dense(input_length / allHead, activation='relu'))

    def call(self, code, input_sentence):
        res=[]
        if self.oriHead>0:
            queuesOri = []
            for i in range(self.oriHead):
                queuesOri.append(self.queueProduceOri[i](code))
            keysOri = []
            for i in range(self.oriHead):
                keysOri.append(tf.transpose(self.keyProduceOri[i](code), perm=[0, 2, 1]))
            attentionsOri = []
            for i in range(self.oriHead):
                attentionsOri.append(
                    tf.reshape(tf.matmul(queuesOri[i], keysOri[i]), shape=(-1, INPUT_SIZE, INPUT_SIZE,1)))
            # 对于ori来说，取三层的最大值
            allAttnOri = tf.concat(attentionsOri, axis=3)
            attenRes = tf.reduce_max(allAttnOri, axis=3)
            sentenceAttentionOri = tf.matmul(input_sentence, attenRes)
            maskedSentAttnOri = input_sentence * sentenceAttentionOri
            res = [tf.matmul(maskedSentAttnOri, code)]
        if self.inMaskHead>0:
            valuesIn = []
            for i in range(self.inMaskHead):
                valuesIn.append(self.valueProduceIn[i](code))
            queuesIn = []
            for i in range(self.inMaskHead):
                queuesIn.append(self.queueProduceIn[i](code))
            keysIn = []
            for i in range(self.inMaskHead):
                keysIn.append(tf.transpose(self.keyProduceIn[i](code), perm=[0, 2, 1]))
            attentionsIn = []
            for i in range(self.inMaskHead):
                attentionsIn.append(tf.matmul(queuesIn[i], keysIn[i]))
            sentenceAttentionsIn = []
            for i in range(self.inMaskHead):
                sentenceAttentionsIn.append(tf.matmul(input_sentence, attentionsIn[i]))
            maskedSentAttnsIn = []
            for i in range(self.inMaskHead):
                maskedSentAttnsIn.append(input_sentence * sentenceAttentionsIn[i])
            for i in range(self.inMaskHead):
                res.append(tf.matmul(maskedSentAttnsIn[i], valuesIn[i]))
        if self.outMaskHead>0:
            valuesOut = []
            for i in range(self.outMaskHead):
                valuesOut.append(self.valueProduceOut[i](code))
            queuesOut = []
            for i in range(self.outMaskHead):
                queuesOut.append(self.queueProduceOut[i](code))
            keysOut = []
            for i in range(self.outMaskHead):
                keysOut.append(tf.transpose(self.keyProduceOut[i](code), perm=[0, 2, 1]))
            attentionsOut = []
            for i in range(self.outMaskHead):
                attentionsOut.append(tf.matmul(queuesOut[i], keysOut[i]))
            sentenceAttentionsOut = []
            for i in range(self.outMaskHead):
                sentenceAttentionsOut.append(tf.matmul(input_sentence, attentionsOut[i]))
            maskedSentAttnsOut = []
            for i in range(self.outMaskHead):
                thisMaskedSentAttnsOut=(input_sentence-1) * sentenceAttentionsOut[i]
                # 用reducesum求出每一句的包含几个token
                senteceInclude = tf.reshape(tf.reduce_sum(input_sentence, axis=2), shape=(-1, SENTENCE_SIZE, 1))
                # 然后求出句外有几个token
                senteceOutclude = 500 - senteceInclude
                # 强度映射
                thisMaskedSentAttnsOut = thisMaskedSentAttnsOut / senteceOutclude * senteceInclude
                maskedSentAttnsOut.append(thisMaskedSentAttnsOut)
            for i in range(self.outMaskHead):
                res.append(tf.matmul(maskedSentAttnsOut[i], valuesOut[i]))
        return res

class mulityHeadAttnGCN(tf.keras.layers.Layer):
    def __init__(self, input_length,head, **kwargs):
        super(mulityHeadAttnGCN, self).__init__(**kwargs)
        self.length = input_length
        self.head=head
        self.value=[]
        self.queue=[]
        self.key=[]
        for i in range(head):
            self.value.append(tf.keras.layers.Dense(input_length/head, activation='relu'))
            self.queue.append(tf.keras.layers.Dense(input_length/head, activation='relu'))
            self.key.append(tf.keras.layers.Dense(input_length/head, activation='relu'))
        self.finalDense=tf.keras.layers.Dense(input_length,activation="relu")
    def call(self, code, input_relation):
        input_relation_trans = tf.transpose(input_relation, perm=[0, 2, 1])
        res=[]
        for i in range(self.head):
            if i<self.head/2:
                this_relation = input_relation
            else:
                this_relation = input_relation_trans
            # x,128,256
            sentenceValue=self.value[i](code)
            # x,128,256
            sentenceQueue = self.queue[i](code)
            # x,128,256
            sentenceKey = self.key[i](code)
            sentenceKeyT = tf.transpose(sentenceKey, perm=[0, 2, 1])
            # x,128,128
            sentenceAttention = tf.matmul(sentenceQueue, sentenceKeyT)

            input_relation = addSelf(sequence_length=128)(this_relation)

            # 获得atten之后，使用relation作mask
            # 为了避免除数为0加一个小数
            # x,128,128
            pdgAttn = this_relation * sentenceAttention + 0.00000000001

            # sum一下
            indegreed = tf.math.sqrt(tf.reduce_sum(pdgAttn, axis=1, keepdims=True))
            outdegreed = tf.math.sqrt(tf.reduce_sum(pdgAttn, axis=2, keepdims=True))

            # softmax一下
            pdgAttn = pdgAttn/indegreed
            pdgAttn = pdgAttn/outdegreed

            softed = this_relation * pdgAttn

            # x,128,256
            res.append(tf.matmul(softed, sentenceValue))

        # x,128,256
        allChannel = tf.concat(res, 2)
        addOri= tf.concat([allChannel,code],2)
        res =self.finalDense(addOri)
        return res

class finalClassifier(tf.keras.layers.Layer):
    def __init__(self, filters1, filters2, kernel_size1, kernel_size2, maxPoolSize, dropOutRate=None, **kwargs):
        super(finalClassifier, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Convolution1D(filters=filters1, kernel_size=(kernel_size1), padding='same',
                                                   activation='relu')
        self.conv2 = tf.keras.layers.Convolution1D(filters=filters2, kernel_size=(kernel_size2), padding='same',
                                                   activation='relu')
        self.maxPoolLayer1 = tf.keras.layers.MaxPool1D(pool_size=maxPoolSize)
        self.maxPoolLayer2 = tf.keras.layers.MaxPool1D(pool_size=maxPoolSize)
        if not dropOutRate == None:
            self.drop1 = tf.keras.layers.Dropout(dropOutRate)
            self.drop2 = tf.keras.layers.Dropout(dropOutRate)
            self.ifDrop = True
        else:
            self.ifDrop = False
        self.flat1 = tf.keras.layers.Flatten()
        self.flat2 = tf.keras.layers.Flatten()
        self.mlp1 = tf.keras.layers.Dense(64, activation='relu')
        self.mlp2 = tf.keras.layers.Dense(16, activation='relu')
        self.outputLayer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, code):
        # x,128,512
        conv1d10 = self.conv1(code)
        conv1d20 = self.conv2(code)
        maxPool1 = self.maxPoolLayer1(conv1d10)
        maxPool2 = self.maxPoolLayer2(conv1d20)
        if self.ifDrop:
            dropOut1 = self.drop1(maxPool1)
            dropOut2 = self.drop2(maxPool2)
            fc1 = self.flat1(dropOut1)
            fc2 = self.flat2(dropOut2)
        else:
            fc1 = self.flat1(maxPool1)
            fc2 = self.flat2(maxPool2)
        fcAll = tf.concat([fc1, fc2], 1)
        dense1 = self.mlp1(fcAll)
        dense2 = self.mlp2(dense1)
        outputs = self.outputLayer(dense2)

        return outputs
def buildModelDMCA():
    # 500 & 128*500 & 128*128
    input_token = tf.keras.Input(shape=(INPUT_SIZE), name="input_token")
    input_sentence = tf.keras.Input(shape=(SENTENCE_SIZE, INPUT_SIZE), name="input_sentence")
    input_relation = tf.keras.Input(shape=(SENTENCE_SIZE, SENTENCE_SIZE), name="input_relation")
    embedding = tf.keras.layers.Embedding(input_dim=WORDS_SIZE,
                                          output_dim=embedding_dim,
                                          weights=[random_weights],
                                          input_length=INPUT_SIZE)(input_token)
    encoder = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True, reset_after=True
                                                                , activation='tanh', recurrent_activation='sigmoid'),name='encoder')(
        embedding)
    # x,500,128
    encoderNorm = tf.keras.layers.BatchNormalization()(encoder)
    res = doubleSideAttnWithOri(input_length=128, oriHeadNum=4, inMaskHeadNum=3, outMaskHeadNum=1)(encoderNorm, input_sentence)

    # 获得句子信息
    # x,128,256
    sentenceCode = tf.concat(res, axis=2)

    input_relation_addself=addSelf(SENTENCE_SIZE)(input_relation)

    res1 = mulityHeadAttnGCN(input_length=256, head=4)(sentenceCode,input_relation_addself)
    res2 = mulityHeadAttnGCN(input_length=256, head=4)(res1,input_relation_addself)

    # x,128,512
    decoder = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True, reset_after=True
                                                                , activation='tanh', recurrent_activation='sigmoid'))(
        res2)


    outputs = finalClassifier(filters1=512, filters2=512, kernel_size1=10, kernel_size2=20, maxPoolSize=4,
                              dropOutRate=0.5)(decoder)

    return tf.keras.Model((input_token, input_sentence, input_relation), outputs)

def change2dict(inputList):
    res={}
    for each in inputList:
        res[each[0]]=each[1]
    return res

def runOneModel(weight,wordModel,analyFolder,kMax):

    model = buildModelDMCA()

    print(model.summary())
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, epsilon=1, decay=0.0,
                                           amsgrad=False)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights(weight)

    attnCountFolder = analyFolder

    attnCountList = os.listdir(attnCountFolder)
    heatmap_model = tf.keras.Model(inputs=[model.get_layer('input_token').input, model.get_layer('input_sentence').input,
                                           model.get_layer('input_relation').input],
                                   outputs=[model.get_layer('encoder').output, model.output])
    allTokenCountor={}
    contributionTokenCountor={}
    totalSample=0
    allTokenCountorRight={}
    contributionTokenCountorRight={}
    totalSampleRight=0
    allTokenCountorFalse={}
    contributionTokenCountorFalse={}
    totalSampleFalse=0
    for each in attnCountList:
        totalSample+=1
        thisFile = pd.read_pickle(attnCountFolder + "/" + each)
        print(each)
        this_token = tf.keras.preprocessing.sequence.pad_sequences([pd.DataFrame(thisFile[0])])
        this_sentence = tf.keras.preprocessing.sequence.pad_sequences([pd.DataFrame(thisFile[1])])
        this_relation = tf.keras.preprocessing.sequence.pad_sequences([pd.DataFrame(thisFile[2])])
        # this_res = tf.keras.preprocessing.sequence.pad_sequences([pd.DataFrame(thisFile[3])])
        with tf.GradientTape() as gtape:
            encoderLayer, prediction = heatmap_model((this_token, this_sentence, this_relation))
            grads = gtape.gradient(prediction, encoderLayer)
            if prediction>0.5:
                totalSampleRight+=1
                flag=True
                print("right")
            else:
                totalSampleFalse+=1
                flag=False
                print("false")
        pooled_grads = np.array(K.mean(grads, axis=(0, 2)))
        for i in range(500):
            if not this_token[0][i][0] > 0:
                pooled_grads[i] = 0
        # the10th = findKthLargest(pooled_grads, 40)
        the10th = findKthLargest(pooled_grads, kMax)
        pooled_grads = np.array(K.mean(grads, axis=(0, 2)))
        for i in range(500):
            if not this_token[0][i][0] > 0:
                pooled_grads[i] = 0
        for i in range(500):
            contributionToken = this_token[0][i][0]
            if contributionToken in allTokenCountor:
                allTokenCountor[contributionToken] += 1
            else:
                allTokenCountor[contributionToken] = 1
            if flag==True:
                if contributionToken in allTokenCountorRight:
                    allTokenCountorRight[contributionToken] += 1
                else:
                    allTokenCountorRight[contributionToken] = 1
            else:
                if contributionToken in allTokenCountorFalse:
                    allTokenCountorFalse[contributionToken] += 1
                else:
                    allTokenCountorFalse[contributionToken] = 1
            if pooled_grads[i] >= the10th and not pooled_grads[i]==0:
                # print(contributionToken)
                if contributionToken<=0: continue
                if contributionToken in contributionTokenCountor:
                    contributionTokenCountor[contributionToken]+=1
                else:
                    contributionTokenCountor[contributionToken]=1
                if flag==True:
                    if contributionToken in contributionTokenCountorRight:
                        contributionTokenCountorRight[contributionToken] += 1
                    else:
                        contributionTokenCountorRight[contributionToken] = 1
                else:
                    if contributionToken in contributionTokenCountorFalse:
                        contributionTokenCountorFalse[contributionToken] += 1
                    else:
                        contributionTokenCountorFalse[contributionToken] = 1
    # contributionTokenCountor=sorted(contributionTokenCountor.items(), key=lambda x: x[1])
    allTokenCountor=sorted(allTokenCountor.items(), key=lambda x: x[1])
    contributionTokenCountorRight=sorted(contributionTokenCountorRight.items(), key=lambda x: x[1])
    allTokenCountorRight=sorted(allTokenCountorRight.items(), key=lambda x: x[1])
    # contributionTokenCountorFalse=sorted(contributionTokenCountorFalse.items(), key=lambda x: x[1])
    # allTokenCountorFalse=sorted(allTokenCountorFalse.items(), key=lambda x: x[1])
    token2vector = Word2Vec.load(wordModel)
    tokenDictOri=token2vector.wv.key_to_index
    tokenDict={}
    for k,v in tokenDictOri.items():
        tokenDict[v]=k
    contribution=contributionTokenCountorRight
    allToken= change2dict(allTokenCountorRight)
    resDict={}
    for each in contribution:
        thisToken = each[0]
        thisTimes = each[1]
        if thisTimes>2:
            tfIdf=thisTimes*1.0/allToken[thisToken]
            resDict[tokenDict[thisToken]]=(thisTimes,tfIdf)
    res=sorted(resDict.items(), key=lambda x: x[1][1])
    for each in res:
        print(each[0],"  ",each[1])
    sum=0
    for each in allTokenCountor:
        if each[0]!=0:
            sum+=each[1]
    print(sum/totalSample)

def parse_options():
    parser = argparse.ArgumentParser(description='Extracting Cpgs.')
    parser.add_argument('-w', '--weight', help='The dir path of model weight', type=str, required=True)
    parser.add_argument('-wm', '--wordModel', help='The dir path of word2vec moedel', type=str, required=True)
    parser.add_argument('-f', '--files', help='The dir path of pkl file to test', type=str, required=True)
    parser.add_argument('-k', '--kMax', help='If the token ranks higher than this number, it is considered to have contributed to this program classification', type=int, default=10)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args=parse_options()
    runOneModel(args.weight,args.wordModel,args.files,args.kMax)

