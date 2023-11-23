import argparse
import os
import pickle
import time

import networkx as nx
import numpy as np
import tensorflow as tf
import pandas as pd
import re
from gensim.models.word2vec import Word2Vec

WORDS_SIZE = 10000
INPUT_SIZE = 500
NUM_CLASSES = 2
MODEL_NUM = 0
EPOCHS = 10
SENTENCE_SIZE = 128
SENTENCE_LENGTH = 256

def buildTokenDictByWord2vect(dirname):
    model = Word2Vec.load(dirname)
    tokenDict = {}
    for k,v in model.wv.key_to_index.items():
        if v<=WORDS_SIZE:
            tokenDict[k] = v
    maxLen = 0
    for k, v in tokenDict.items():
        if len(k) > maxLen:
            maxLen = len(k)
    return tokenDict,maxLen

def convert(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def my_tokenizer(code):
    ## Remove code comments
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    code = re.sub(pat,'',code)
    ## Remove newlines & tabs
    code = re.sub('(\n)|(\\\\n)|(\\\\)|(\\t)|(/)|(\\r)','',code)
    ## Mix split (characters and words)
    splitter = '\"(.*?)\"| +|(;)|(->)|(&)|(\*)|(\()|(==)|(~)|(!=)|(<=)|(>=)|(!)|(\+\+)|(--)|(\))|(=)|(\+)|(\-)|(\[)|(\])|(<)|(>)|(\.)|({)'
    code = re.split(splitter,code)
    ## Remove None type
    code = list(filter(None, code))
    code = list(filter(str.strip, code))
    # snakecase -> camelcase and split camelcase
    code_1 = []
    for i in code:
        code_1 += convert(i).split('_')
    #filt
    code_2 = []
    for i in code_1:
        if i in ['{', '}', ';', ':']:
            continue
        code_2.append(i)
    return(code_2)

def processOneByLine(tokenDict,dotDataFolder,cDataFolder,outFolder):
    dotFiles=os.listdir(dotDataFolder)
    outFiles=os.listdir(outFolder)
    count=0
    for file in dotFiles:
        print(count/len(dotFiles))
        count+=1
        try:
            if file.replace(".dot",".pkl") in outFiles:
                continue
            dot = dotDataFolder+"/"+file
            # print("read "+dot)
            # import pdg
            pdg = nx.drawing.nx_pydot.read_dot(dot)
            # get source code
            codeFile=cDataFolder+"/"+file.replace(".dot",".c")
            source_ids = [-1]
            sentence = np.zeros((SENTENCE_SIZE, INPUT_SIZE))
            tokensNumLast = 1
            f = open(codeFile)
            contents = f.read().split("\n")
            for i in range(min(len(contents), SENTENCE_SIZE)):
                content = contents[i]
                if content.find("static void") != -1:
                    content = content.replace("static void", "void")
                tokens = my_tokenizer(content)
                for eachToken in tokens:
                    if eachToken in tokenDict:
                        source_ids.append(tokenDict[eachToken])
                tokensNumNow = len(source_ids)
                if tokensNumNow >= INPUT_SIZE:
                    tokensNumNow = INPUT_SIZE - 1
                for pos in range(tokensNumLast, tokensNumNow):
                    sentence[i][pos] = 1
                if tokensNumNow == INPUT_SIZE - 1:
                    break
                tokensNumLast = tokensNumNow
            tokenLength = len(source_ids)
            if tokenLength >= INPUT_SIZE:
                # source_tokens = source_tokens[:INPUT_SIZE]
                source_ids = source_ids[:INPUT_SIZE - 1] + [-2]
            else:
                # source_tokens = source_tokens + [''] * (INPUT_SIZE - tokenLength)
                source_ids = source_ids + [-2] + [0] * (INPUT_SIZE - tokenLength - 1)

            labels_dict = nx.get_node_attributes(pdg, 'label')
            label_lineDict = {}
            for label, all_code in labels_dict.items():
                # "10"[label = "<(&lt;operator&gt;.assignment,VAR1 = -1)<SUB>5</SUB>>"]
                lineNum = int(all_code[all_code.index("<SUB>") + 5:all_code.index("</SUB>")])
                label_lineDict[int(label)] = lineNum

            pdgEdges = nx.get_edge_attributes(pdg, 'label')
            sentenceRela = []
            for relation, label in pdgEdges.items():
                # ('18', '146', 0) "DDG: VAR7 = VAR8"
                inline = label_lineDict[int(relation[0])]
                outline = label_lineDict[int(relation[1])]
                sentenceRela.append([inline, outline])
            indegreed = np.zeros((SENTENCE_SIZE, SENTENCE_SIZE))

            for each in sentenceRela:
                # print(each)
                inSent = each[0]
                outSent = each[1]
                if inSent < SENTENCE_SIZE and outSent < SENTENCE_SIZE:
                    indegreed[inSent-1][outSent-1]=1

            out_pkl = outFolder+"/" + file.replace(".dot",".pkl")
            data = [source_ids, sentence, indegreed]
            with open(out_pkl, 'wb') as f:
                pickle.dump(data, f)

        except Exception as e:
            print(str(e))

def parse_options():
    parser = argparse.ArgumentParser(description='Extracting Cpgs.')
    parser.add_argument('-d', '--dotFile', help='The dir path of dot file', type=str, required=True)
    parser.add_argument('-c', '--cFile', help='The dir path of source code', type=str, required=True)
    parser.add_argument('-o', '--output', help='The dir path of output', type=str, required=True)
    parser.add_argument('-m', '--model', help='The dir path of word2vec model', type=str, required=True)
    args = parser.parse_args()
    return args
def main():
    args = parse_options()
    model = Word2Vec.load(args.model)
    processOneByLine(model.wv.key_to_index,args.dotFile,args.cFile,args.output)

if __name__ == '__main__':
    main()
