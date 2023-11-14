import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat


def stat_char(data):
    mean_var = data.mean()
    median = data.median()
    disp = data.var()
    standart_var = data.std()
    plt.hist(data, 23)
    print("Математическое ожидание: ", mean_var)
    print("Медиана: ", median)
    print("Дисперсия: ", disp)
    print("Среднеквадратическое отклонение: ", standart_var)


def stat_hip(data):
    mean = data.mean()
    std = data.std()
    size = 100
    data = np.random.choice(data, size, replace=False)
    res_pears = stat.normaltest(data)
    res_ks = stat.kstest(data, "norm", N=size, args=(mean, std))
    res_crammis = stat.cramervonmises(data, 'norm', args=(mean, std))
    res_shapiro = stat.shapiro(data)
    res_jb = stat.jarque_bera(data)
    res_and = stat.anderson(data)
    print("Уровни значимости")
    print("Критерий Пирсона: ", res_pears.pvalue)
    print("Критерий Колмогорова: ", res_ks.pvalue)
    print("Критерий Крамера-Мизеса: ", res_crammis.pvalue)
    print("Критерий Шапиро-Уилка", res_shapiro.pvalue)
    print("Критерий Харке-Бера: ", res_jb.pvalue)
    print("Критерий Андерсона-Дарлинга: ", res_and.statistic)
    print("Критические значения: ", res_and.critical_values)
    print("Уровни значимости: ", res_and.significance_level)
    norm_data = stat.norm.rvs(loc=mean, scale=std, size=size)
    plt.figure()
    plt.hist(data, 15, label="Исходные данные")
    plt.hist(norm_data, 15, label="Нормальное распределение")



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
    x_norm = np.arange(0, 100, 1)
    y_norm = stat.norm.pdf(x_norm, loc=50, scale=10)
    plt.figure()
    plt.plot(x_norm, y_norm)
    stat_hip(height)
    plt.xlabel("Рост, см")
    plt.ylabel("Количество, чел")
    plt.title("Распределение роста людей")
    plt.legend()
    plt.show()
