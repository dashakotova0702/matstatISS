from library import calc

def arithmetic_operation(a, b, c):
    print(a + b)  # сумма
    print(b - c)  # разность
    print(a / c)  # деление
    print(a * b)  # умножение
    print(c // b)  # деление без остатка
    print(b ** c)  # возведение в степень
    print(pow(b, c))  # возведение в степень
    print(c % 2)  # остаток от деления (на 2)
    print(abs(b))  # модуль
    print(a < b)  # сравнение
    print(c > b)  # сравнение
    print(b == a)  # проверка на равенство
    print(a != c)  # проверка на неравенство
    print(b < a and c != b)  # сравнение и проверка на неравенство
    print(b > c or a == c)  # сравнение или проверка на равенство
    print(not a)  # отрицание


if __name__ == "__main__":
    a = 1                   # целое число
    b = -2.5                # вещественное число
    c = 3.0                 # тоже вещественное
    arithmetic_operation(a, b, c)
    calc(c)
