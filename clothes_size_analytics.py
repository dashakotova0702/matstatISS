import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat


def stat_char(data):
    mean_var = data.mean()
    median = data.median()
    disp = data.var()
    standart_var = data.std()
    plt.hist(data, 50)
    print("Математическое ожидание: ", mean_var)
    print("Медиана: ", median)
    print("Дисперсия: ", disp)
    print("Среднеквадратическое отклонение: ", standart_var)


def stat_hip(data):
    mean = data.mean()
    std = data.std()
    size = data.size
    norm_data = stat.norm.rvs(loc=mean, scale=std, size=size)
    res_pears = stat.normaltest(data)
    res_ks = stat.kstest(data, "norm", N=data.size, args=(mean, std))
    res_shapiro = stat.shapiro(roll_data)
    res_jb = stat.jarque_bera(roll_data)
    res_and = stat.anderson(roll_data)
    print("Уровни значимости")
    print("Критерий Пирсона: ", res_pears.pvalue)
    print("Критерий Колмогорова: ", res_ks.pvalue)
    print("Критерий Шапиро-Уилка", res_shapiro.pvalue)
    print("Критерий Харке-Бера: ", res_jb.pvalue)
    print("Критерий Андерсона-Дарлинга: ", res_and.statistic)
    print("Критические значения: ", res_and.critical_values)
    print("Уровни значимости: ", res_and.significance_level)



if __name__ == "__main__":
    data = pd.read_csv("clothes_size.csv")

    def filter_nan(x):
        return not np.isnan(x)

    data = data[data['height'].apply(filter_nan)]
    height = data.loc[:, 'height']
    age = data.loc[:, 'age']
    stat_char(height)
    plt.xlabel("Рост, см")
    plt.ylabel("Количество, чел")
    plt.title("Распределение роста людей")
    plt.figure()
    stat_char(age)
    plt.xlabel("Возраст, г")
    plt.ylabel("Количество, чел")
    plt.title("Распределение возраста людей")
    stat_hip(height)
    plt.show()
