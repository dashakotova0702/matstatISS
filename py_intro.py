a = 1                   # целое число
b = -2.5                # вещественное число
c = 3.0                 # тоже вещественное
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

list_1 = []  # создаем пустой лист
list_1.insert(0, 1)  # добавляем целое число на 1 место (индекс 0)
list_1.append(2)  # добавляем целое число в конец списка
list_1.extend([3, 4])  # добавляем набор элементов в конец списка
list_1.remove(4)  # удаляем элемент из списка
list_1.insert(5, 5)  # добавляем целое число на 6 место (индекс 5)
list_1.append("cat")  # добавляем строку в конец списка
list_1.append(["dog", "pig"])  # добавляем список строк в конец списка
print(list_1)

list_2 = [1, 2, 3, 5, "cat", ["dog", "pig"]]  # тоже самое, но сразу
print(list_2)

print(list_1 + list_2)  # объединение списков
print(list_1 == list_2)  # сравнение списков
print(len(list_1))  # длина списка
print(list_2[3])  # обращение к элементу по индексу
print(list_2[2:4])  # срез (последний элемент не включительно)
print(list_2[:3])  # срез от начала списка

d = range(6)  # диапазон от 0 до 6 с шагом 1
e = range(0, 6, 1)  # диапазон от 0 до 6 с шагом 1
f = range(10, 5, -1)  # диапазон от 10 до 5 с шагом -1
g = range(0, 1000, 100)  # диапазон от 0 до 1000 с шагом 100

for i in d:  # цикл вывода диапазона
    print(i)

for i in e:  # цикл вывода диапазона
    print(i)

for i in range(5, 0, -1):  # цикл вывода диапазона
    print(i)

for i in range(1, 10, 1):  # цикл проверки диапазона чисел на четность
    if i % 2:
        print("odd")
    else:
        print("even")

stroka = "Hello World"  # задаем строку
print(len(stroka))  # размер строки
print(stroka[5])  # 6-ой символ строки
print(stroka + '!')  # сложение строк
print(stroka * 2)
print(stroka.split())  # разделить строку по пробелу

for symb in stroka:
    print(symb)

while c > 0:        # цикл уменьшения числа
    c -= 1
print(c)