# -*- coding:utf-8 -*-
import MeCab
import os
import logging
from gensim import corpora, models

class TextMining(object):
    def __init__(self):
        self.tagger = MeCab.Tagger('-Ochasen')
        self.target_features = ['名詞']
        self.path = './Documents/test/'
        self.get_words_from_text('')

    def node_generator(self, text):
        node = self.tagger.parseToNode(text)
        while node:
            if node.feature.split(',')[0] in self.target_features:
                yield node.surface.lower()
            node = node.next

    def get_text_from_documents(self):
        textlist = []
        filelist = os.listdir(self.path)
        for filename in filelist:
            if os.path.isfile(self.path + filename):
                with open(self.path + filename) as f:
                    textlist.append(self.get_words_from_text(f.read()))
        return textlist

    def get_words_from_text(self, text):
        return [word for word in self.node_generator(text)]

class Dictionary(object):
    def __init__(self):
        self._data = None

    def create_data(self, text):
        #data.filter_extremes(no_below = no_below, no_above = no_above)    #no_below => 上限 no_above => 下限
        return corpora.Dictionary(texts)

    def set_data(self, dictionary):
        self._data = dictionary

    def get_data(self):
        return self._data

    def save(self, filename='dictionary'):
        self._data.save_as_text('./%s.txt'%filename)

    data = property(get_data, set_data)

class Corpus(object):
    def __init__(self):
        self._data = None

    def create_data(self, dictionary, texts):
        return [dictionary.doc2bow(text) for text in texts]

    def set_data(self, corpus):
        self._data = corpus

    def get_data(self):
        return self._data

    def save(self, filename='./corpus'):
        corpora.MmCorpus.serialize('./%s.mm'%filename, self._data)

    data = property(get_data, set_data)

class Tfidf(object):
    def __init__(self):
        self._data = None

    def create_data(self, corpus):
        return models.TfidfModel(corpus)

    def set_data(self, tfidf):
        self._data = tfidf

    def get_data(self):
        return self._data

    def save(self, filepath='./tfidf_corpus'):
        self._data.save(filepath)

    data = property(get_data, set_data)

class Lda(object):
    def __init__(self):
        self._data = None

    def create_data(self, bow, id2word, topic=300):
        return models.LdaModel(bow, id2word=id2word, num_topics=topic)

    def set_data(self, lda):
        self._data = lda

    def get_data(self):
        return self._data

    def save(self, filepath='./lda_model'):
        self._data.save(filepath)

    data = property(get_data, set_data)

if __name__ == '__main__':
    textmining = TextMining()
    texts = textmining.get_text_from_documents()

    dictionary = Dictionary()
    dictionary_data = dictionary.create_data(texts)
    dictionary.data = dictionary_data

    corpus = Corpus()
    corpus_data = corpus.create_data(dictionary.data, texts)
    corpus.data = corpus_data

    tfidf = Tfidf()
    tfidf_data = tfidf.create_data(corpus.data)
    tfidf.set_data(tfidf_data)

    bow = [dictionary.data.doc2bow(text) for text in texts]

    lda = Lda()
    lda_data = lda.create_data(bow, dictionary.data, 300)
    lda.set_data(lda_data)

    dictionary.save()
    corpus.save()
    tfidf.save()
    lda.save()

#    entitie = [dictionary, corpus, tfidf, lda]
#    map(lambda x:x.save(), entitie)
