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
    data_sel = np.random.choice(data, size, replace=False)
    res_pears = stat.normaltest(data_sel)
    res_ks = stat.kstest(data_sel, "norm", N=size, args=(mean, std))
    res_crammis = stat.cramervonmises(data_sel, 'norm', args=(mean, std))
    res_shapiro = stat.shapiro(data_sel)
    res_jb = stat.jarque_bera(data_sel)
    res_and = stat.anderson(data_sel)
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
    res_stud = stat.ttest_1samp(data_sel, mean)
    data_2 = np.random.choice(data, size, replace=False)
    res_stud_2 = stat.ttest_ind(data_sel, data_2)
    res_ks_2 = stat.kstest(data_sel, data_2, N=size, args=(mean, std))
    print("Критерий Стьюдента (сравнение с мат.ожиданием): ", res_stud.pvalue)
    print("Критерий Стьюдента (сравнение 2 выборок): ", res_stud_2.pvalue)
    print("Критерий Смирнова: ", res_ks_2.pvalue)
    plt.figure()
    plt.hist(data_sel, 15, label="Первая выборка")
    plt.hist(norm_data, 15, label="Нормальное распределение")
    plt.hist(data_2, 15, label="Вторая выборка")



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
