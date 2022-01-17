# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 19:19:59 2021

@author: 1952640
"""

import numpy as np
import tensorflow.keras as kr
import pandas as pd
import random

def news_process(news):
    content=[]
    entity=[]
    for i in range(len(news)):
        content_one=news.title[i].strip('[').strip(']').split(',')
        content_one.extend(news.abstract[i].strip('[').strip(']').split(','))
        entity_one=news.title_entities[i].strip('[').strip(']').split(',')
        entity_one.extend(news.abstract_entities[i].strip('[').strip(']').split(','))
        content.append(list(map(int,content_one)))
        entity.append(list(map(int,entity_one)))
    news_content=dict(zip(list(news.id),content))
    news_entity=dict(zip(list(news.id),entity))
    return news_content,content,news_entity,entity

def batch_iter(users,news_id,news,news_entity,entity,batch_size=64,max_length=100,click_size=30,candidate_size=10,real_num=3):
    batch_count=0
    click_batch=[]
    candidate_batch=[]
    click_entity_batch=[]
    candidate_entity_batch=[]
    for i in range(len(users)):
        if(batch_count==batch_size):
            batch_count=0
            click_batch=[]
            candidate_batch=[]
            click_entity_batch=[]
            candidate_entity_batch=[]
        click_id=str(users.history.iloc[i]).split(" ")
        click=[]
        click_entity=[]
        for j in range(len(click_id)):
            if(click_id[j] in news_id):
                click.append(news_id[click_id[j]])
                click_entity.append(news_entity[click_id[j]])
            else:
                j-=1
        if(len(click)<candidate_size+real_num):
            continue
        batch_count+=1
        candidate=click[-real_num:]
        candidate_entity=click_entity[-real_num:]
        for j in range(candidate_size-real_num):
            randno=random.randint(0,120958)
            candidate.append(news[randno])
            candidate_entity.append(entity[randno])
            
        click= kr.preprocessing.sequence.pad_sequences(click[0:click_size],max_length)
        click_entity= kr.preprocessing.sequence.pad_sequences(click_entity[0:click_size],max_length)
        if(len(click)<click_size):
            pad=np.zeros(shape=(click_size-len(click),max_length),dtype=np.int)
            click=np.concatenate((click,pad))
            click_entity=np.concatenate((click_entity,pad))
        click_batch.append(click)
        click_entity_batch.append(click_entity)
        candidate_batch.append( kr.preprocessing.sequence.pad_sequences(candidate,max_length))
        candidate_entity_batch.append( kr.preprocessing.sequence.pad_sequences(candidate_entity,max_length))
        if(batch_count==batch_size):
            yield click_batch,click_entity_batch,candidate_batch,candidate_entity_batch,candidate_batch[0]
    return
        

        