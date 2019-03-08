#encoding=utf-8
#Author: CJ
#Date: 2019-02-26 15:42:49

import os
import json
'''
标注格式 BIOES
查看是否有嵌套的实体
'''

DATA_ROOT_DIR = 'data/CCKS2017/CCKS2017_dataset/case_of_illness/data2/training dataset v4'
EFFECTIVE_DIR = ('病史特点','出院情况','一般项目','诊疗经过')
TAG_MAP = {'身体部位':'BODY', '症状和体征':'SYMPTOM','疾病和诊断':'DISEASE','检查和检验':'CHECK', '治疗':'CUE'}

def tagging():
    tagged_set = []
    char_count = 0
    for dir_ in os.listdir(DATA_ROOT_DIR):
        if dir_ in EFFECTIVE_DIR:
            for index in range(1,301):
                file1 = DATA_ROOT_DIR+'/'+dir_+'/'+dir_+'-'+str(index)+'.txt'
                file2 = DATA_ROOT_DIR+'/'+dir_+'/'+dir_+'-'+str(index)+'.txtoriginal.txt'

                if not os.path.exists(file1):
                    continue

                with open(file1, 'r') as f1:
                    content1 = f1.read().strip('\n').split('\n')
                
                if content1[0] == '':
                    continue

                with open(file2, 'r') as f2:
                    content2 = f2.read().decode('utf8').strip()

                content1_count = 0
                begin, end = int(content1[content1_count].split('\t')[1]), int(content1[content1_count].split('\t')[2])
                tag = content1[content1_count].split('\t')[3].strip('\r')
                for i, c in enumerate(content2):
                    char_count += 1 
                    if i < begin or i > end:
                        tagged_set.append(c+'\t'+'O\n')
                    elif begin <= i <= end:
                        if begin == i < end:
                            tagged_set.append(c+'\t'+'B-'+TAG_MAP[tag]+'\n')
                        elif begin == i and end == i:
                            tagged_set.append(c+'\t'+'S-'+TAG_MAP[tag]+'\n')
                        elif i < end:
                            tagged_set.append(c+'\t'+'I-'+TAG_MAP[tag]+'\n')
                        elif i == end:
                            tagged_set.append(c+'\t'+'E-'+TAG_MAP[tag]+'\n')
                            content1_count += 1
                            if content1_count != len(content1):
                                begin, end = int(content1[content1_count].split('\t')[1]), int(content1[content1_count].split('\t')[2])
                                tag = content1[content1_count].split('\t')[3].strip('\r')
                tagged_set.append('\n\r')

    with open('tagged_data.txt', 'w') as output_f:
        for item in tagged_set:
            output_f.write(item)
    print char_count
    
def char_to_idx():
    with open('tagged_data.txt', 'r') as f:
        tagged_set = f.read().split('\n')
    char_set = []
    for row in tagged_set:
        char = row.split('\t')[0]
        char_set.append(char)
    char_set = set(char_set)
    char_json = {}
    for idx, c in enumerate(list(char_set)):
    # for idx, c in enumerate(char_set):
        if c == '':
            continue
        char_json[c] = idx-1
    
    with open('char2idx.json', 'w') as j:
        json.dump(char_json, j)

    with open('charset.txt', 'w') as f:
        for char in char_set:
            f.write(char+'\n')

def tag_to_idx():
    with open('tagged_data.txt', 'r') as f:
        tagged_set = f.read().split('\n')
    tag_set = []
    for row in tagged_set:
        try:
            tag = row.split('\t')[1]
            tag_set.append(tag)
        except:
            continue
    tag_set = set(tag_set)
    tag_json = {}
    for idx, c in enumerate(tag_set):
        if c == '':
            continue
        tag_json[c] = idx-1
    with open('tag2idx.json', 'w') as j:
        json.dump(tag_json, j)

if __name__ == "__main__":
    tagging()