import argparse

import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn.metrics

from models import doubleSideAttnWithOri,addSelf,mulityHeadAttnGCN,finalClassifier

INPUT_SIZE = 500
WORDS_SIZE = 5000
SENTENCE_SIZE = 128
SENTENCE_LENGTH = 256
embedding_dim = 32
# 随机初始化
np.random.seed(20237)
random_weights = np.random.normal(size=(WORDS_SIZE, embedding_dim), scale=0.01)

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
                                                                , activation='tanh', recurrent_activation='sigmoid'))(
        embedding)
    # x,500,128
    encoderNorm = tf.keras.layers.BatchNormalization()(encoder)
    res = doubleSideAttnWithOri(input_length=128, oriHeadNum=4, inMaskHeadNum=3, outMaskHeadNum=1)(encoderNorm, input_sentence)

    # 获得句子信息
    # x,128,256
    sentenceCode = tf.concat(res, axis=2)

    input_relation=addSelf(SENTENCE_SIZE)(input_relation)

    res1 = mulityHeadAttnGCN(input_length=256, head=4)(sentenceCode,input_relation)
    res2 = mulityHeadAttnGCN(input_length=256, head=4)(res1,input_relation)

    # x,128,512
    decoder = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True, reset_after=True
                                                                , activation='tanh', recurrent_activation='sigmoid'))(
        res2)


    outputs = finalClassifier(filters1=512, filters2=512, kernel_size1=10, kernel_size2=20, maxPoolSize=4,
                              dropOutRate=0.5)(decoder)

    return tf.keras.Model((input_token, input_sentence, input_relation), outputs)

def calResult(y_real,predicted_prob):
    predicted = np.int64(predicted_prob > 0.5)
    confusion = sklearn.metrics.confusion_matrix(y_true=y_real, y_pred=predicted)

    print(confusion)
    tn, fp, fn, tp = confusion.ravel()
    print('\nTP:', tp)
    print('FP:', fp)
    print('TN:', tn)
    print('FN:', fn)

    # 假阳性
    FPR = fp / (fp + tn)
    # 假阴性
    FNR = fn / (tp + fn)
    print('FPR:', FPR)
    print('FNR:', FNR)

    ## Performance measure
    print('\nAccuracy: ' + str(sklearn.metrics.accuracy_score(y_true=y_real, y_pred=predicted)))
    print('Precision: ' + str(sklearn.metrics.precision_score(y_true=y_real, y_pred=predicted)))
    print('Recall: ' + str(sklearn.metrics.recall_score(y_true=y_real, y_pred=predicted)))
    print('F-measure: ' + str(sklearn.metrics.f1_score(y_true=y_real, y_pred=predicted)))
    print('Precision-Recall AUC: ' + str(
        sklearn.metrics.average_precision_score(y_true=y_real, y_score=predicted_prob)))
    print('AUC: ' + str(sklearn.metrics.roc_auc_score(y_true=y_real, y_score=predicted_prob)))
    print('MCC: ' + str(sklearn.metrics.matthews_corrcoef(y_true=y_real, y_pred=predicted)))

def runOneModel(buildfun,trainpath="./dataset/train.pickle",valpath="./dataset/val.pickle",testpath="./dataset/test.pickle",modelSavePath="./model",modelLoadPath="./model/model-COMB-100-single.hdf5",test=False,epoch=0):
    if test==False:
        # 读取数据
        train = pd.read_pickle(trainpath)
        # 将输入分成token和矩阵两组
        x_train_token = tf.keras.preprocessing.sequence.pad_sequences(train['codeVector'])
        x_train_sentence = tf.keras.preprocessing.sequence.pad_sequences(train['sentenceArray'])
        x_train_pdg = tf.keras.preprocessing.sequence.pad_sequences(train['indegreeArray'])
        y_train = train['isvul'].astype(np.int64)
        del train

        val = pd.read_pickle(valpath)
        # 看看数据是否正常

        print(val.iloc[0:5, 0:])

        x_val_token = tf.keras.preprocessing.sequence.pad_sequences(val['codeVector'])#.astype(np.int64)
        x_val_sentence = tf.keras.preprocessing.sequence.pad_sequences(val['sentenceArray'])  # .astype(np.int64)
        x_val_pdg = tf.keras.preprocessing.sequence.pad_sequences(val['indegreeArray'])  # .astype(np.int64)
        y_val = val['isvul'].astype(np.int64)
        del val


        model = buildfun()
        print(model.summary())

        adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, epsilon=1, decay=0.0,
                                               amsgrad=False)

        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

        # print(model.summary())

        callbackdir = modelSavePath+'/cb'
        tbCallback = tf.keras.callbacks.TensorBoard(log_dir=callbackdir,
                                                    histogram_freq=5,
                                                    embeddings_freq=5,
                                                    write_graph=True,
                                                    write_images=True)
        tbCallback.set_model(model)
        mld = modelSavePath+'/model-COMB-{epoch:02d}-single.hdf5'

        ## Create best model callback
        mcp = tf.keras.callbacks.ModelCheckpoint(filepath=mld,
                                                 monitor="val_accuracy",
                                                 save_best_only=True,
                                                 mode='auto',
                                                 save_freq='epoch',
                                                 verbose=1)

        # 100epoch
        try:
            history = model.fit(x=(x_train_token, x_train_sentence, x_train_pdg),
                                y=y_train,
                                validation_data=((x_val_token, x_val_sentence, x_val_pdg), y_val),
                                epochs=epoch,
                                batch_size=256,
                                verbose=2,
                                callbacks=[mcp, tbCallback])
        except Exception:
            pass

        del x_train_token, x_train_sentence, x_train_pdg, y_train
        del x_val_token, x_val_sentence, x_val_pdg, y_val

        test = pd.read_pickle(testpath)
        x_test_token = tf.keras.preprocessing.sequence.pad_sequences(test['codeVector']).astype(np.int64)
        x_test_sentence = tf.keras.preprocessing.sequence.pad_sequences(test['sentenceArray'])  # .astype(np.int64)
        x_test_pdg = tf.keras.preprocessing.sequence.pad_sequences(test['indegreeArray'])  # .astype(np.int64)
        y_test = test['isvul'].astype(np.int64)
        del test

        predicted_prob = model.predict((x_test_token,x_test_sentence, x_test_pdg))
        calResult(y_test,predicted_prob)

    else:
        model = buildfun()
        model.load_weights(modelLoadPath)
        real = pd.read_pickle(testpath)
        x_real_token = tf.keras.preprocessing.sequence.pad_sequences(real['codeVector']).astype(np.int64)
        x_real_sentence = tf.keras.preprocessing.sequence.pad_sequences(real['sentenceArray'])  # .astype(np.int64)
        x_real_pdg = tf.keras.preprocessing.sequence.pad_sequences(real['indegreeArray'])  # .astype(np.int64)
        y_real = real['isvul'].astype(np.int64)
        del real

        predicted_prob = model.predict((x_real_token, x_real_sentence, x_real_pdg))
        calResult(y_real, predicted_prob)

def parse_options():
    parser = argparse.ArgumentParser(description='Extracting Cpgs.')
    parser.add_argument('-tr', '--train', help='The dir path of train dataset', type=str)
    parser.add_argument('-v', '--val', help='The dir path of train dataset', type=str)
    parser.add_argument('-te', '--test', help='The dir path of train dataset', type=str, required=True)
    parser.add_argument('-wm', '--workMode', help='The train or only test', type=str, required=True, default="train")
    parser.add_argument('-s', '--savePath', help='The dir path for save model', type=str)
    parser.add_argument('-l', '--loadPath', help='The dir path for load model', type=str)
    parser.add_argument('-e', '--epoch', help='The train epoch', type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_options()
    if args.workMode=="train":
        runOneModel(buildModelDMCA,trainpath=args.train,valpath=args.val,testpath=args.test,modelSavePath=args.savePath,test=False,epoch=args.epoch)
    elif args.workMode=="test":
        runOneModel(buildModelDMCA,testpath=args.test,modelLoadPath=args.loadPath,test=True)
    else:
        print("No such mode")


if __name__ == "__main__":
    main()
