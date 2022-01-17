# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:02:24 2020

@author: 1952640
"""

from collections import Counter
import numpy as np
import tensorflow.keras as kr
import os

# 对两个版本的py的处�?
'''
if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False
'''

#�?0个作为已点击�?
#�?个候�?其中一个正样本
#取最后一�?一个假的为测试�?

def open_file(filename,mode='r'):
    #print(os.getcwd())
    #open('../data/val.csv', mode='w').write('\n')
    return open(filename,mode,encoding='utf-8',errors='ignore')#

# 读取文件数据 返回 内容+标签 列表的列�?
def read_file(filename):
    # 列表！列表！不是数组
    contents, users, newsids = [], [], []
    with open_file(filename) as f:
        print(f)
        for line in f:
           # print('?')
            #try:
            if 1:
                #print(line.strip().split(','))
                # 将每一行的元素变为list，strip()删除的字�?按照split()中的符号进行每行元素分割为list的元�?
                this_line=line.strip().split(',')
                user="".join(this_line[0])
                newid=this_line[1]
                #print(newid)
                cltime=this_line[2]
                title=this_line[3]
                content=this_line[4]
                #print('ok')
                #print(user)
                #print(title)
                if user:
                    users.append(user)
                if newid:
                    newsids.append(list(newid))
                if title:#如果不是空的
                   # print(list(title))
                    cont=list(title)
                    cont.extend(list(content))
                    contents.append(cont)
                    #contents.append(list(content))#�?list_new 看作一个对象，整体打包添加�?list 对象中�?
                    #labels.append(label)
            #except:
                #pass
    return contents, users, newsids

# 根据训练集构建词汇表，存�?参数1：训练集文件 2：词汇表储存文件
def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    # 内容与标�?
    data_train, _, __ = read_file(train_dir)
    all_data = []
    #print(data_train)
    for content in data_train:
        # �?list_new 看作一个序列，将这个序列和 list 序列合并，并放在其后面�?
        #print(content)
        all_data.extend(content)
    counter = Counter(all_data)
    # 对相同单词进行计数，个数为vocab_size - 1
    # 返回值为[(xx,1),(xxx,2)]
    count_pairs = counter.most_common(vocab_size - 1)
    #print(data_train)
    #print(count_pairs)
    # *将元�?不可更改)解压为列�?
    # zip将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表�?
    # zip() 返回的是一个对象。如需展示列表，需手动 list() 转换�?
    words, _ = list(zip(*count_pairs))#这里是解压缩 words是词�?_是计�?
    # 添加一�?<PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    # 写入词汇表文�?
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')

# 从词汇表文件读取词汇�?
def read_vocab(vocab_dir):
    with open_file(vocab_dir) as fp:
        words = [_.strip() for _ in fp.readlines()]
    #把词汇表转化为字�?
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    categories = [x for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id

# 将id表示的内容转换为文字
# word[id]=词汇
def to_words(content, words):
    return ''.join(words[x] for x in content)#生成一个字符串

# 将文件转换为id 即文字变为数字列�?
# 字典的访�?id['name']
# 返回内容
# 词向量会在encoder里生成，这里不需要生成词向量
def process_file(filename, word_to_id, cat_to_id, max_length=600):
    contents, users, newsids = read_file(filename)
    data_id = []
    for i in range(len(contents)):
        # 将该条新闻的每个词转化为id
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        #label_id.append(cat_to_id[labels[i]])# 把该条新闻的类型转为id
    # 使用keras提供的pad_sequences来将文本pad为固定长�?
    news = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    # 将标签转换为one-hot表示
    # y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  
    '''
    新闻标题数量*类别
    [1,0,0,0]
    [0,1,0,0]
    ....
    '''
    return news,users #, y_pad

# 生成批次数据
# 以一个用户为一个训练批�?
# 取前70%作为训练集，取此数据开�?0条为候选新�?
# 返回训练集，候选新闻，确实被浏览的新闻
def batch_iter(news,users,max_length=600,candidate_num=5,click_num=20,batch_size=64):
    user_count=Counter(users)
    i=0
    #第一条是预测的正样本
    tot_news=len(news)
    while (i<tot_news):
        batch_click,batch_candidate,batch_real=[],[],[]
        #pad=[0]*max_length
        j=0
        while(j<batch_size):
            click=int(user_count[users[i]]-2)
            if(click>click_num):
                click=click_num
            if(click<=3):
                i+=user_count[users[i]]
                continue
            input_click=news[i+1:i+click+1]
            input_real=news[i:i+1]
            #print(input_click.shape)
            if(click<click_num):
                pad=np.zeros(shape=(click_num-click,input_click.shape[-1]),dtype=np.int)
                input_click=np.concatenate((input_click,pad))
                #input_click[click_num-1][0]=-1
                #input_click[click_num-1][1]=click
                click=click_num
                #print(input_click.shape)
            #append(input_real)
            #print('--')
            #print(i+user_count[users[i]])
            #print(i+user_count[users[i]]+candidate_num-1)
            #print('--')
            if(i+user_count[users[i]]+candidate_num-1>tot_news):
                return
            input_candidate=np.concatenate((input_real,news[i+user_count[users[i]]:i+user_count[users[i]]+candidate_num-1]))
            #print(input_click.shape)
            #print(input_candidate.shape)

            if (i+click_num+candidate_num)<tot_news:
                batch_click.append(input_click)
                batch_candidate.append(input_candidate)
                batch_real.append(input_real)
                j+=1
                #yield input_click,input_candidate,input_real 
                #yield news[i:i+click_num],news[i+click_num+1:i+click_num+candidate_num],news[i+click_num+1:i+user_count[users[i]]]
            else:
                return 
            i+=user_count[users[i]]
        yield batch_click,batch_candidate,batch_real

def test_process_file(filename, word_to_id, cat_to_id, max_length=600):
    contents, users, newsids = read_file(filename)
    data_id = []
    for i in range(len(contents)):
        # 将该条新闻的每个词转化为id
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        #label_id.append(cat_to_id[labels[i]])# 把该条新闻的类型转为id
    # 使用keras提供的pad_sequences来将文本pad为固定长�?
    news = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    # 将标签转换为one-hot表示
    # y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  
    '''
    新闻标题数量*类别
    [1,0,0,0]
    [0,1,0,0]
    ....
    '''
    return news,users,contents #, y_pad
            
def test_batch_iter(news,users,max_length=600,candidate_num=5,click_num=20,batch_size=64):
    user_count=Counter(users)
    i=0
    #第一条是预测的正样本
    tot_news=len(news)
    while (i<tot_news):
        batch_click,batch_candidate,batch_real,userno=[],[],[],[]
        #pad=[0]*max_length
        j=0
        while(j<batch_size):
            no=[]
            click=int(user_count[users[i]]-2)
            if(click>click_num):
                click=click_num
            if(click<=3):
                i+=user_count[users[i]]
                continue
            input_click=news[i+1:i+click+1]
            input_real=news[i:i+1]
            #print(input_click.shape)
            if(click<click_num):
                pad=np.zeros(shape=(click_num-click,input_click.shape[-1]),dtype=np.int)
                input_click=np.concatenate((input_click,pad))
                #input_click[click_num-1][0]=-1
                #input_click[click_num-1][1]=click
                click=click_num
                #print(input_click.shape)
            #append(input_real)
            #print('--')
            #print(i+user_count[users[i]])
            #print(i+user_count[users[i]]+candidate_num-1)
            #print('--')
            if(i+user_count[users[i]]+candidate_num-1>tot_news):
                return
            input_candidate=np.concatenate((input_real,news[i+user_count[users[i]]:i+user_count[users[i]]+candidate_num-1]))
            #print(input_click.shape)
            #print(input_candidate.shape)
            no.extend([i+1,i+1+click+1,i,i+user_count[users[i]],i+user_count[users[i]]+candidate_num-1])
            if (i+click_num+candidate_num)<tot_news:
                batch_click.append(input_click)
                batch_candidate.append(input_candidate)
                batch_real.append(input_real)
                userno.append(no)
                j+=1
                #yield input_click,input_candidate,input_real 
                #yield news[i:i+click_num],news[i+click_num+1:i+click_num+candidate_num],news[i+click_num+1:i+user_count[users[i]]]
            else:
                return 
            i+=user_count[users[i]]
        yield batch_click,batch_candidate,batch_real,nolist
            
