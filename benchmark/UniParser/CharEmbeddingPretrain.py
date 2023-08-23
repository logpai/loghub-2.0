# -*- coding: utf-8 -*-
import gensim
from gensim.models import word2vec, FastText
from glove import Glove, Corpus
import h5py
import shutil

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

"""

"""



# ——————————————————————————————————————构建用于预训练的语料———————————————————————————————————————
def createSentenceList(mode):
    files = ['Android', 'Apache', 'BGL', 'HDFS', 'HPC', 'Hadoop', 'HealthApp', 'Linux', 'Mac', 'OpenSSH', 'OpenStack',
             'Proxifier', 'Spark', 'Thunderbird', 'Windows', 'Zookeeper']
    sentenceList = []
    if mode == 'word':
        for i in files:
            with open(f'./inputs/logs/{i}/{i}_2k.log', 'r', encoding='utf-8') as fp:
                lines = fp.readlines()
                sentenceList.extend([l.strip().split(' ') for l in lines])
    elif mode == 'char':
        for i in files:
            with open(f'./inputs/logs/{i}/{i}_2k.log', 'r', encoding='utf-8') as fp:
                lines = fp.readlines()
                for l in lines:
                    sentenceList.extend([[_c for _c in word] for word in l.strip().split(' ') if len(word)>0])
    print(f"[DATA] number of lines: {len(sentenceList)}")
    print(sentenceList[:5])
    return sentenceList


# ——————————————————————————————————————DaGuan杯word2vec,glove,fastText预训练———————————————————————————————————————
def fastTextTrans(model):
    modelList = []
    for key in model.wv.vocab.keys():
        temp = []
        temp.append(key)
        temp += model[key].tolist()
        modelList.append(temp)

    return modelList


def gloveTrans(model):
    modelList = []
    for key in model.dictionary.keys():
        temp = []
        temp.append(key)
        temp += model.word_vectors[model.dictionary[key]].tolist()
        modelList.append(temp)

    return modelList


def getPreTrain(mode, dimNum, minCount, saveFileWord2Vec, saveFileFastText, saveFileGlove):
    # 从语料中构建bichar形式的list，并预训练bichar的word2vec，glove，FastText
    sentenceList = createSentenceList(mode)

    gloveCorpusModel = Corpus()
    gloveCorpusModel.fit(sentenceList, window=10, ignore_missing=False)
    # corpus_model.save('corpus.model')
    print('Dict size: %s' % len(gloveCorpusModel.dictionary))
    print('Collocations: %s' % gloveCorpusModel.matrix.nnz)
    GloveModel = Glove(no_components=dimNum, learning_rate=0.05)
    GloveModel.fit(gloveCorpusModel.matrix, epochs=25, no_threads=20, verbose=True)
    GloveModel.add_dictionary(gloveCorpusModel.dictionary)
    glove = gloveTrans(GloveModel)
    with open(saveFileGlove, 'w', encoding='utf-8') as fr1:
        for temp in glove:
            s = ''
            for ch in temp:
                s = s + str(ch) + ' '
            fr1.writelines(s.strip() + '\n')
    fr1.close()

    # 上下文窗口默认为5
    Word2VecModel = gensim.models.Word2Vec(sentenceList, size=dimNum, sg=1, iter=15, window=10, workers=20,
                                           min_count=minCount)
    Word2VecModel.wv.save_word2vec_format(saveFileWord2Vec, binary=False)

    FastTextModel = FastText(sentenceList, size=dimNum, window=10, workers=20, sg=1, iter=15, min_n=2, max_n=8,
                             min_count=minCount)
    fastText = fastTextTrans(FastTextModel)
    with open(saveFileFastText, 'w', encoding='utf-8') as fr2:
        for temp in fastText:
            s = ''
            for ch in temp:
                s = s + str(ch) + ' '
            fr2.write(s.strip() + '\n')
    fr2.close()


def tokenEmbeddingFileToEmbedding(hdf5Path, savePath):
    fr = h5py.File(hdf5Path, 'r')
    fr1 = open(savePath, 'w', encoding='utf-8')
    fr2 = open('inputs/corpus/DaGuan/DaGuanVocabForElmo.txt', 'r', encoding='utf-8')
    tokenList = fr2.readlines()
    for i in range(len(fr['embedding'])):
        embedding = fr['embedding'][i].tolist()
        embedding = list(map(str, embedding))
        token = tokenList[i].strip()
        fr1.write(token + ' ')
        fr1.write(' '.join(embedding) + '\n')


# ——————————————————————————————————————构建所需语料———————————————————————————————————————
# constructCorpusForDaGuan()

# ——————————————————————————————————————150dim word2Vec,fastText预训练———————————————————————————————————————
print("Training word embedding 100")
getPreTrain(mode='word', dimNum=100, minCount=5,
            saveFileWord2Vec='inputs/embedding_matrix/char_log/Log2kWord_Word2Vec_100dim.txt',
            saveFileFastText='inputs/embedding_matrix/char_log/Log2kWord_FastText_100dim.txt',
            saveFileGlove='inputs/embedding_matrix/char_log/Log2kWord_Glove_100dim.txt')
print("Training char embedding 100")
getPreTrain(mode='char', dimNum=100, minCount=5,
            saveFileWord2Vec='inputs/embedding_matrix/char_log/Log2kChar_Word2Vec_100dim.txt',
            saveFileFastText='inputs/embedding_matrix/char_log/Log2kChar_FastText_100dim.txt',
            saveFileGlove='inputs/embedding_matrix/char_log/Log2kChar_Glove_100dim.txt')
