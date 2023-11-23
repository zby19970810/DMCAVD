import tensorflow as tf
import numpy as np
from gensim.models.word2vec import Word2Vec

INPUT_SIZE = 500
SENTENCE_SIZE = 128
class embeddingWithWord2Vec(tf.keras.layers.Layer):
    def __init__(self, dirname, **kwargs):
        super(embeddingWithWord2Vec, self).__init__(**kwargs)
        model = Word2Vec.load(dirname)
        embeddingDict=[]
        for k,v in model.wv.key_to_index.items():
            if v<=3882:
                embeddingDict.append(model.wv[k])
        self.embeddingDict=tf.convert_to_tensor(embeddingDict)
    def call(self, inputs):
        onehot=tf.one_hot(inputs,3882)
        res = tf.matmul(onehot,self.embeddingDict)
        return res
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