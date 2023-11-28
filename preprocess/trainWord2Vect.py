## coding: utf-8
'''
This python file is used to tranfer the words in corpus to vector, and save the word2vec model.
'''

from gensim.models.word2vec import Word2Vec
import argparse
import os
import gc
import re

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

class DirofCorpus(object):
    def __init__(self, dirname):
        self.dirname = dirname
        tokenedSample=[]
        for dirs in self.dirname:
            for files in os.listdir(dirs):
                if files == ".keep" or files.endswith(".txt") or files.endswith(".csv"):
                    continue
                print(files)
                try:
                    samples = open(os.path.join(dirs, files), 'r')
                    tokenedSample.append(my_tokenizer(samples.read()))
                    del samples
                    gc.collect()
                except Exception as e:
                    print(str(e))
        self.allSentenceTokenedSample=tokenedSample

    def __iter__(self):
        for sample in self.allSentenceTokenedSample:
            yield sample


def generate_w2vModel(decTokenFlawPath, w2vModelPath,word_dim=128):
    print("training...")
    model = Word2Vec(sentences=DirofCorpus(decTokenFlawPath), vector_size=word_dim, alpha=0.01, window=5, min_count=8,
                     max_vocab_size=10000, sample=0.001, seed=1, workers=1, min_alpha=0.0001, sg=1, hs=0, negative=10,
                     epochs=9)
    model.save(w2vModelPath)


def evaluate_w2vModel(w2vModelPath):
    print("\nevaluating...")
    model = Word2Vec.load(w2vModelPath)
    for sign in ['void', '=', '(', '[', 'main']:
        print(sign, ":")
        print(model.predict_output_word(context_words_list=[sign], topn=10))

def parse_options():
    parser = argparse.ArgumentParser(description='Extracting Cpgs.')
    parser.add_argument('-i', '--input', help='The dir path of input', type=str, required=True)
    parser.add_argument('-o', '--output', help='The dir path of output', type=str, required=True)
    parser.add_argument('-d', '--dim', help='The type of procedures: parse or export', type=int, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_options()
    generate_w2vModel([args.input], args.output, word_dim=args.dim)
    evaluate_w2vModel(args.output)
    print("success!")


if __name__ == "__main__":
    main()
