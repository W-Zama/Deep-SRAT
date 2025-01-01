import pandas as pd


class Dataset():
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)

    def get_data(self):
        return self.df

    def get_columns(self):
        return self.df.columns

    def get_column(self, column_name):
        return self.df[column_name]

    def get_column_values(self, column_name):
        return self.df[column_name].values

    def get_column_names(self):
        return self.df.columns

    def get_column_name(self, index):
        return self.df.columns[index]

    def get_column_index(self, column_name):
        return self.df.columns.get_loc(column_name)

    def set_column_name(self, testing_date_column_name, num_of_failures_per_unit_time_column_name):
        # もし，カラム名が空文字列の場合はエラーを出す
        if not testing_date_column_name or not num_of_failures_per_unit_time_column_name:
            raise ValueError("Column name cannot be empty.")
        else:
            self.testing_date_column_name = testing_date_column_name
            self.num_of_failures_per_unit_time_column_name = num_of_failures_per_unit_time_column_name

    def set_dataset(self):
        self.testing_date_df = self.df[[self.testing_date_column_name]]
        self.num_of_failures_per_unit_time_df = self.df[[
            self.num_of_failures_per_unit_time_column_name]]
        self.calc_and_set_cumulative_data()

    def get_testing_date_df(self):
        return self.testing_date_df

    def get_num_of_failures_per_unit_time_df(self):
        return self.num_of_failures_per_unit_time_df

    def calc_and_set_cumulative_data(self):
        self.cumulative_num_of_failures_df = self.num_of_failures_per_unit_time_df.cumsum()
