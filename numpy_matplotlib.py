import numpy as np
import matplotlib.pyplot as plt

# создать массив с единицами произвольного размера
ones_array = np.ones(5)

# создать массив с рандомными числами такого же размера, не вводя явно
rand_array = np.random.random(ones_array.size)

# сложить, вычесть, умножить (скалярно и матрично) массивы и вывести результат
print(ones_array+rand_array)
print(ones_array-rand_array)
print(ones_array*rand_array)
print(ones_array @ rand_array)

# Простые графики
lin = np.linspace(-5, 5, 10)
plt.plot(lin, lin**3, color="red", linestyle="-", label="График кубической функции")
plt.scatter(lin, lin**3, color="black", label="Поточечный график кубической функции")
plt.plot(lin, lin**2, ":b", label="График квадратичной функции")
plt.legend()
plt.title("Простые графики")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(0, 0, "o")
plt.text(-0.5, 5, "(0, 0)")
plt.grid()

# задать матрицу 10*10 рандомными числами, построить график 3 строки,
# 7 столбца, контурный график всей матрицы subplots
matrix = np.random.random((10, 10))
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].plot(matrix[3], color="red")
ax[1].plot(matrix[:, 7], color="blue")
ax[2].contourf(matrix, levels=np.linspace(matrix.min(), matrix.max(), 20))
plt.figure()

# построить 3д график гиперболического параболоида, контурный график, пометить на контурном графике макс и мин точку

x = np.arange(-5, 5, 0.01)
y = np.arange(-5, 5, 0.01)
X, Y = np.meshgrid(x, y)
Z = 0.5*(X**2-Y**2)

arg_max_el = np.unravel_index(np.argmax(Z, axis=None), Z.shape)
arg_min_el = np.unravel_index(np.argmin(Z, axis=None), Z.shape)

ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, Z, cmap="jet")

plt.figure()
plt.contourf(Z, levels=np.linspace(Z[arg_min_el], Z[arg_max_el], 20))
plt.plot(arg_max_el[0], arg_max_el[1], 'ro')
plt.plot(arg_min_el[0], arg_min_el[1], 'bo')
plt.text(arg_max_el[0]*1.05, arg_max_el[1]+30, "({}, {})".format(arg_max_el[0], arg_max_el[1]))
plt.text(arg_min_el[0]+30, arg_min_el[1]*1.05, "({}, {})".format(arg_min_el[0], arg_min_el[1]))
plt.figure()

# сохранить массивы, прочитать и вывести график
with open('hiperboloid.txt', 'wb') as f:
    np.save(f, X)
    np.save(f, Y)
    np.save(f, Z)

with open('hiperboloid.txt', 'rb') as f:
    x_read = np.load(f)
    y_read = np.load(f)
    z_read = np.load(f)
    plt.contourf(z_read, levels=np.linspace(z_read[arg_min_el], z_read[arg_max_el], 20))
plt.figure()
# np.where и чистка NaN

array = np.array([[1, np.nan, -3, 4, 10], [np.nan, -3, np.nan, 6, -5]])
array[np.where(array < 0)] = 0
array[np.isnan(array)] = 0
print(array)

# гистограмма нормального распределения
norm = np.random.normal(0, 2, 1000)
plt.hist(norm, bins=50)

# чтение картинок
plt.imshow(plt.imread('cat.jpg'))
plt.figure()

# отрисовка картинки с рандомными RGB
img = np.random.randint(0, 255, (1000, 1000, 3))
plt.imshow(img)


plt.show()

