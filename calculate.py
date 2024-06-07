import pandas as pd
import json
from tqdm import tqdm

class MAP:
    def __init__(self, main_path):

        self.PREDICT_FILE: str = main_path + "\\result.csv"  # файл с предсказаниями рекомендаций
        self.TEST_LOG_FILE: str = main_path + "\\test.csv"  # файл с тестовой выборкой
        self.TRAIN_LOG_FILE: str = main_path + "\\train.csv"  # файл с обучающей выборкой
        self.LIMIT_ROW: int = 0  # лимит проверенных строчек

    def check_user_predict_data(self, row: pd.Series) -> float:
        number_predict: int = 1
        tp: int = 1
        user_result: int = 0

        train_log_movies: list = self.train_logs_df[
            self.train_logs_df['user_id'] == row['user_id']
            ]['movie_id'].values[0]

        test_logs_movies: list = self.test_logs_df[
            self.test_logs_df['user_id'] == row['user_id']
            ]['movie_id'].values[0]

        predict_movies: list = json.loads(row['predict'])

        for predict_movie in predict_movies:
            if predict_movie in test_logs_movies:
                if predict_movie not in train_log_movies:
                    user_result += tp / number_predict
                    tp += 1
            number_predict += 1
        return user_result / len(predict_movies)

    def calculate(self):
        print('START calculate')
        predict_df: pd.DataFrame = pd.read_csv(
            self.PREDICT_FILE,
            names=['user_id', 'predict']
        )

        self.train_logs_df: pd.DataFrame = pd.read_csv(self.TRAIN_LOG_FILE, names=['user_id', 'movie_id'])
        self.train_logs_df.movie_id = self.train_logs_df.movie_id.apply(eval)
        self.test_logs_df: pd.DataFrame = pd.read_csv(self.TEST_LOG_FILE, names=['user_id', 'movie_id'])
        self.test_logs_df.movie_id = self.test_logs_df.movie_id.apply(eval)

        user_num: int = 0
        list_result: list = []
        index: int
        predict_row: pd.Series
        for index, predict_row in tqdm(predict_df.iterrows(), total=predict_df.shape[0]):
            user_num += 1
            if user_num > self.LIMIT_ROW and self.LIMIT_ROW:
                break
            list_result.append(self.check_user_predict_data(predict_row))
        result_mectic_value: float = sum(list_result) / len(self.train_logs_df['user_id'].value_counts())
        print('\nFINISH calculate')
        print(f'RESULT: {round(result_mectic_value, 8)}')