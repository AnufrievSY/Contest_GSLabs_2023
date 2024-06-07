import re
import numpy as np
from pymorphy3 import MorphAnalyzer
from collections import Counter
from itertools import chain
from sklearn.preprocessing import MinMaxScaler
import torch
from nltk.corpus import stopwords as nltk_stopwords



class MyLemmatizer:
  def __init__(self):
    self.norm_dict = {}
    self.word_dict = {}
    self.morph = MorphAnalyzer()
    self.stopwords = tuple(nltk_stopwords.words('russian'))

  def prepare(self, text: str) -> tuple:
    prep_text = text.lower()
    prep_text = re.sub(r'\b\w\b', '', prep_text)
    prep_text = re.sub(r'\b\w\w\b', '', prep_text)
    prep_text = re.sub(r'[^\w\s]', ' ', prep_text)
    prep_text = (i for i in prep_text.split() if i != '')
    prep_text = (i for i in prep_text if i not in self.stopwords)
    return tuple(prep_text)

  def fit(self, texts:list):
    old_words = self.prepare(' '.join(texts))
    norm_words = (
        max(((r.score, r.normal_form) for r in self.morph.parse(word)), key=lambda x: x[0])[1]
        for word in old_words
    )
    self.norm_dict = dict(zip(old_words, norm_words))
    self.word_dict = dict(Counter(old_words))
    max_count = max(self.word_dict.values())
    self.word_dict = {word: count / max_count for word, count in self.word_dict.items()}

  def transform(self, texts:list):
    if texts:
      prep_texts = self.prepare(texts)
      if prep_texts:

        return np.mean(
                  [self.word_dict.get(
                      self.norm_dict.get(word, 0.0),
                      self.word_dict.get(word, 0.0))
                  for word in prep_texts])
      else:
        return 0.0
    else:
      return 0.0


class MyUserPreferenceClassifier :
  def __init__(self):
    self.items_dict = {}

  def fit(self, items:list):
    items = list(chain.from_iterable(items))
    c = Counter(items)
    self.items_dict = {k:v/max(c.values()) for k, v in c.items()}

  def transform(self, user_items:list):
    if user_items:
      return np.mean(
                [self.items_dict[item]
                 if item in self.items_dict.keys()
                 else 0.0
                 for item in user_items])
    else:
      return 0.0


class DataPrepare:
  def __init__(self, id:int, databases):
    self.user_id = id
    self.logs = databases['logs']
    self.movies = databases['movies']
    self.movies = self.movies.drop(['id', 'name', 'year', 'date_publication'], axis=1)
    self.scaler = MinMaxScaler()
    self.attr_count = MyUserPreferenceClassifier()
    self.lematizer = MyLemmatizer()
    self.movies_id = self.logs.loc[self.logs.user_id==self.user_id].movie_id.unique().astype(int)


  def transform(self):
    t = self.movies.loc[self.movies_id, 'description'].tolist()

    self.lematizer.fit(t)
    self.movies.description = self.movies.description.apply(self.lematizer.transform)

    self.attr_count.fit(self.movies.loc[self.movies_id, 'genres'].tolist())
    self.movies.genres = self.movies.genres.apply(self.attr_count.transform)

    self.attr_count.fit(self.movies.loc[self.movies_id, 'countries'].tolist())
    self.movies.countries = self.movies.countries.apply(self.attr_count.transform)

    self.attr_count.fit(self.movies.loc[self.movies_id, 'staff'].tolist())
    self.movies.staff = self.movies.staff.apply(self.attr_count.transform)

    self.movies['duration'] = 0.0
    for index in self.movies_id:
      self.movies.loc[index, 'duration'] = sum(self.logs.loc[self.logs.user_id == self.user_id].loc[self.logs.movie_id == index].duration)
    x = torch.tensor(self.movies.iloc[:, :-1].values, dtype=torch.float32)
    y = np.array(self.movies.duration).reshape(-1, 1)
    y = self.scaler.fit_transform(y)
    y = torch.tensor(y, dtype=torch.float32)
    return (x, y)