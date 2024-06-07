import os
import warnings
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import data_prepare
import calculate

warnings.filterwarnings("ignore")


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = F.normalize(x, dim=0)
        return x


class MyModel:
    def __init__(self,
                 path: str = None):
        if path:
            self.main_path = path
        else:
            self.main_path = os.getcwd()

        # Загружаем необходимые для работы файлы
        self.logs = pd.read_csv(self.main_path + '\\logs.csv')
        self.genres = pd.read_csv(self.main_path + '\\genres.csv')
        self.movies = pd.read_csv(self.main_path + '\\movies.csv')
        self.staff = pd.read_csv(self.main_path + '\\staff.csv')
        self.countries = pd.read_csv(self.main_path + '\\countries.csv')

        # str -> list
        self.movies['genres'] = self.movies['genres'].apply(eval)
        self.movies['countries'] = self.movies['countries'].apply(eval)
        self.movies['staff'] = self.movies['staff'].apply(eval)

        # Заполняем все пропущенные значения
        self.movies.description = self.movies.description.fillna('')

        self.results = {'train_id': pd.DataFrame(columns=['user_id', 'values']),
                        'test_id': pd.DataFrame(columns=['user_id', 'values']),
                        'predict_id': pd.DataFrame(columns=['user_id', 'values'])}

    def train(self, user_id=None):
        if type(user_id) == list or type(user_id) == tuple:
            self.user_id = user_id
        elif type(user_id) == float:
            self.user_id = self.logs.user_id.unique().tolist()
            self.user_id = self.user_id[:round(len(self.user_id) * user_id)]
        elif type(user_id) == int:
            self.user_id = [user_id]
        elif not user_id:
            self.user_id = self.logs.user_id.unique().tolist()

        self.models = dict()
        for user_id in tqdm(self.user_id):
            try:
                model = Model()
                loss_func = nn.MSELoss()  # средняя квадратичная функция потерь
                optimizer = optim.Adam(model.parameters(), lr=0.01)  # адаптивная оценка моментов

                user_movies = self.logs.loc[self.logs.user_id == user_id].movie_id.unique().astype(int)
                if len(user_movies) < 5:
                    pass
                else:
                    not_user_movies = [index for index in self.movies.index.tolist() if index not in user_movies]
                    train_user_movies, test_user_movies = train_test_split(user_movies, test_size=0.2)
                    self.results['train_id'].loc[len(self.results['train_id'])] = [user_id, list(train_user_movies)]
                    self.results['test_id'].loc[len(self.results['test_id'])] = [user_id, list(test_user_movies)]

                    prep = data_prepare.DataPrepare(user_id, {'logs': self.logs, 'movies': self.movies})
                    data = prep.transform()

                    train_id = list(not_user_movies) + list(train_user_movies)
                    test_id = list(not_user_movies) + list(test_user_movies)

                    n_epoch = 200

                    for epoch in range(n_epoch):
                        # Обнуление градиентов
                        optimizer.zero_grad()
                        # Расчет выходных данных
                        output = model(data[0][train_id])
                        # Расчет потерь
                        loss = loss_func(output, data[1][train_id])
                        # Обратное распространение ошибки
                        loss.backward()
                        # Обновление весов и смещения
                        optimizer.step()

                    predict = model(data[0][test_id])
                    results = self.movies.loc[test_id].copy()
                    results['duration'] = predict.detach().numpy()
                    results = results.iloc[:, [0, -1]]
                    results = results.loc[results.duration >= results.describe().loc['75%', 'duration']]
                    results = results.sort_values(by='duration', ascending=False)

                    self.results['predict_id'].loc[len(self.results['predict_id'])] = [user_id, results.index.tolist()]

                    self.results['train_id'].to_csv('train.csv', header=False)
                    self.results['test_id'].to_csv('test.csv', header=False)
                    self.results['predict_id'].to_csv('result.csv', header=False)
            except Exception as error:
                print(f'{user_id}: {error}')


model = MyModel('C:/Users/Пользователь/Desktop/contest2023/')
model.train()

calc = calculate.MAP(os.getcwd())
calc.calculate()
