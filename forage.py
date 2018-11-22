'''
    This is the file used to process the text corpus
'''
import jieba
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np

class Forage:
    def __init__(self, corpus, stop=True, stop_type=1, custom_stop_dict=[], init_mode=0):
        '''
        This function is used to initialize some values for Forage class.
        :param corpus: the corpus that you want to train with.
        :param custom_stop_dict: the word that you want to added into stop words, comes in the type of list
        :param inti_mode: the mode that
        :param stop_type: if value is 1 then load the complete default Chinese stop words dict, which you can find in the data folder called stopWords_complete.txt
        if value is 0 then load another default stop words which only contains some common special symbols that make little difference to the text
        and you can find it in the data folder called stopWords_noCharacter.txt.
        :param custom_stop_dict: append the custom stop words after default stop words.
        '''
        self.subjects = []
        self.df_corpus = pd.DataFrame(columns=['content_id', 'content', 'subject'])
        self.corpus = corpus
        self.custom_stop_dict = []
        self.stop = stop
        self.stop_type = stop_type
        self.init_mode = init_mode

        # data for training
        self.corpus_x = []
        self.corpus_y = []
        self.label_encoder = LabelEncoder()

        # inti data
        self.intiCoprus()
        # pre process data
        self.preProcessCorpus()
        # characterize corpus
        self.characterizeCorpus()

    def intiCoprus(self):
        '''
        This function is used to initialize corpus, turn the dict data into type of data frame.
        :return:
        '''


        if not isinstance(self.corpus, dict):
            raise ValueError('Invalid corpus type, type dict is needed', self.corpus)
        try:
            # rewrite using load_from_dict method
            dict_temp = {}
            index = 0
            for key, item in self.corpus.items():
                cur_items = item
                # turn the corpus into the form of data frame
                for c in cur_items:
                    dict_temp[index] = [key, c]
                    index += 1
                    # self.df_corpus.loc[len(self.df_corpus)] = (len(self.df_corpus), c, key)
            self.df_corpus = pd.DataFrame.from_dict(dict_temp, orient='index', columns=['subject', 'content'])
        except ValueError as e:
            print(e)
        finally:
            print('Load corpus completed! \n')

    def preProcessCorpus(self):
        '''
        Pre process the data using jieba and filter stop words according to the stop words dict.
        :return:
        '''

        # load stop words
        stop_words = []
        file_path = 'data/stopWords_{0}.txt'.format(('complete' if self.stop_type == 1 else 'noCharacter'))
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            f.close()
        for l in lines:
            stop_words.append(l.strip('\n').strip())
        # append custom stop words dict
        try:
            stop_words = stop_words + self.custom_stop_dict
        except ValueError as e:
            print(e, 'type list is needed', self.custom_stop_dict)

        # cut sentence and gen
        def toolFun(x):
            x_cut = jieba.cut(x)
            res = ' '.join(filter(lambda x: x not in stop_words, x_cut))
            return res

        self.df_corpus['processed_content'] = self.df_corpus['content'].map(toolFun)
        print('Pre process corpus complete! \n')


    def characterizeCorpus(self):
        '''
        Tokenize the text data both in tf-idf and bag of words level
        !!! Remain to do: tf-idf in word level should be more suitable for most cases, remain to realize
        :return:
        '''
        tf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer='char')
        ha_vectorizer = HashingVectorizer(ngram_range=(1, 1), lowercase=False)

        if self.init_mode == 0:
            # default inti
            tf_idf = tf_vectorizer.fit_transform(self.df_corpus['processed_content'])
            ha = ha_vectorizer.fit_transform(self.df_corpus['processed_content'])
            res = hstack([tf_idf, ha]).tocsr()
        elif self.init_mode == 1:
            # only use tf-idf
            res = (tf_vectorizer.fit_transform(self.df_corpus['processed_content'])).tocsr()
        elif self.init_mode == 2:
            res = (ha_vectorizer.fit_transform(self.df_corpus['processed_content'])).tocsr()
        self.corpus_x = res

        # label encoder for labels
        self.label_encoder.fit(self.df_corpus['subject'].unique())
        self.corpus_y = self.label_encoder.transform(self.df_corpus['subject'].values)

        print('Tokenizing complete! \n')

