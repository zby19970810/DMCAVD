# coding=utf-8
import os
import re
import shutil
import argparse
from clean_gadget import clean_gadget
from timeout_decorator import timeout

def parse_options():
    parser = argparse.ArgumentParser(description='Normalization.')
    parser.add_argument('-i', '--input', help='The dir path of input dataset', type=str, required=True)
    args = parser.parse_args()
    return args

def normalize(path):
    setfolderlist = os.listdir(path)
    for setfolder in setfolderlist:
        catefolderlist = os.listdir(path + "//" + setfolder)
        for catefolder in catefolderlist:
            filepath = path + "//" + setfolder + "//" + catefolder
            print(catefolder)
            try:
                pro_one_file(filepath)
            except Exception as e:
                print(str(e))

def lineNormOneFile(code):
    countForSmall=0
    countForMid=0
    newCode=""
    for each in code:
        if each=="(": countForSmall+=1
        elif each=="[": countForMid+=1
        elif each==")": countForSmall-=1
        elif each=="]": countForMid-=1
        if not (countForSmall==0 and countForMid==0):
            if each!="\n":
                newCode+=each
        else:
            newCode += each
    return newCode

@timeout(10)
def pro_one_file(filepath):
    with open(filepath, "r") as file:
        code = file.read()
    file.close()
    code = re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code)
    code = lineNormOneFile(code)
    with open(filepath, "w") as file:
        file.write(code.strip())
    file.close()

    with open(filepath, "r") as file:
        org_code = file.readlines()
        nor_code = clean_gadget(org_code)
    file.close()
    with open(filepath, "w") as file:
        file.writelines(nor_code)
    file.close()

def main():
    args = parse_options()
    normalize(args.input)

if __name__ == '__main__':
    main()
 