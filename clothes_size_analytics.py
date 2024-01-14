import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat
import seaborn as sns
import scipy.optimize as opt
from sklearn.metrics import r2_score


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



def corr(data_1, data_2):
    res_pears = stat.pearsonr(data_1, data_2)
    res_spear = stat.spearmanr(data_1, data_2)
    res_kend = stat.kendalltau(data_1, data_2)
    print("Коэффициент корреляции Пирсона: ", res_pears.statistic)
    print("Коэффициент корреляции Спирмена: ", res_spear.correlation)
    print("Коэффициент корреляции Кендалла: ", res_kend.correlation)
    sns.scatterplot(data=pd.concat([data_1, data_2],axis=1), x=data_1.name, y=data_2.name)


def lin_reg(data_1, data_2):
    res = stat.linregress(data_1, data_2)
    plt.plot(data_1, data_2, '.', label="Исходные данные")
    plt.plot(data_1, res.intercept + res.slope*data_1, label="Регрессия")
    plt.legend()
    print("Коэффициент детерминации для линейной регрессии: ", res.rvalue**2)
    print("Предсказание на 2024 год: ", res.intercept + res.slope*2024)


def multilin_reg(data_1, data_2, data_3):
    def func(x, a, b, c):
        return a + b * x[0] + c * x[1]
    popt, pcov = opt.curve_fit(func, np.array([data_1, data_2]), data_3)
    pred_data_3 = func(np.array([data_1, data_2]), *popt)
    print("Коэффициент детерминации для множественной линейной регрессии", r2_score(pred_data_3, data_3))
    ax = plt.axes(projection="3d")
    ax.plot(data_1, data_2, data_3, '.', label="Исходные данные")
    d1, d2 = np.meshgrid(data_1, data_2)
    d3 = func(np.array([d1, d2]), *popt)
    ax.plot_wireframe(d1, d2, d3, color='orange', label="Регрессия")
    print("Предсказание на 2024 год: ", func(np.array([2024, 45]), *popt))


def nonlin_reg(data_1, data_2):
    def func(x, a, b, c):
        return a * x + b * x ** 2 + c
    popt, pcov = opt.curve_fit(func, data_1, data_2)
    pred_data_2 = func(data_1, *popt)
    print("Коэффициент детерминации для нелинейной регрессии", r2_score(pred_data_2, data_2))
    plt.plot(data_1, data_2, '.', label="Исходные данные")
    d1 = np.arange(np.min(data_1), np.max(data_1), 0.1)
    d2 = func(d1, *popt)
    plt.plot(d1, d2)
    print("Предсказание на 2024 год: ", func(2024, *popt))



if __name__ == "__main__":
    data = pd.read_csv("clothes_size.csv")
    data = data.dropna()
    height = data.loc[:, 'height']
    age = data.loc[:, 'age']
    print("Статистические характеристики для данных о росте людей\n")
    stat_char(height)
    plt.xlabel("Рост, см")
    plt.ylabel("Количество, чел")
    plt.title("Распределение роста людей")
    plt.figure()
    print("\n\nСтатистические характеристики для данных о возрасте людей\n")
    stat_char(age)
    plt.xlabel("Возраст, г")
    plt.ylabel("Количество, чел")
    plt.title("Распределение возраста людей")
    x_norm = np.arange(0, 100, 1)
    y_norm = stat.norm.pdf(x_norm, loc=50, scale=10)
    plt.figure()
    plt.plot(x_norm, y_norm)
    print("\n\nПроверка гипотез о нормальности данных о росте людей\n")
    stat_hip(height)
    plt.xlabel("Рост, см")
    plt.ylabel("Количество, чел")
    plt.title("Распределение роста людей")
    plt.legend()
    print("\n\nПроверка гипотез о нормальности данных о возрасте людей\n")
    stat_hip(age)
    plt.xlabel("Возраст, г")
    plt.ylabel("Количество, чел")
    plt.title("Распределение возраста людей")
    plt.legend()
    size_num = 40
    uniq_size = ['XXS', 'XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL']
    for size_let in uniq_size:
        data.loc[data['size'] == size_let, 'size'] = size_num
        size_num += 2
    data['size'] = pd.to_numeric(data['size'])
    plt.figure()
    print("\n\nРасчет коэффициентов корреляции между весом и размером одежды людей\n")
    corr(data['size'], data['weight'])
    plt.figure()
    sns.heatmap(data.corr(), annot=True)
    plt.figure()
    sns.scatterplot(data[['age', 'weight']], x='age', y='weight')
    plt.figure()
    print("\n\nРасчет коэффициентов корреляции между весом и ростом людей\n")
    corr(data['height'], data['weight'])
    plt.figure()
    sns.scatterplot(data[['age', 'height']], x='age', y='height')
    print("\n\nКоэффициенты детерминации регресионных моделей\n")
    year = np.array([2019, 2020, 2021, 2022, 2023])
    money = np.array([20, 10, 25, 30, 40])
    happy = np.array([3, 2, 3, 4, 5])
    plt.figure()
    lin_reg(year, happy)
    plt.figure()
    multilin_reg(year, money, happy)
    plt.figure()
    nonlin_reg(year, happy)
    plt.show()