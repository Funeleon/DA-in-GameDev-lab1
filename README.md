# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #1 выполнил(а):
- Баскаков Данил Викторович
- РИ-211102
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Ознакомиться с основными операторами зыка Python на примере реализации линейной регрессии.

## Задание 1
### Написать программы Hello World на Python и Unity. 

Работа на пайтоне:
- Зайти на сайт https://colab.research.google.com 
- Написать и запустить программу "Hello World"
![Снимок](https://user-images.githubusercontent.com/114385414/192873298-3e7a2dd8-514d-433b-ba61-cfca991c006b.PNG)
- Сохранить файл и найти его в облаке
![image](https://user-images.githubusercontent.com/114385414/192873827-83a23fbc-76e5-468c-b254-9217402f47e7.png)

Работа на юнити:
- Отрыть юнити
- Создать новый проект
- Загрузить старый, Т. К. новый не создаётся по какой-то причине
- Создать c# скрипт, написать в функции "старт" программу для "Hello World" 
- Закинуть скрипт на камеру и фиксировать результаты
![image](https://user-images.githubusercontent.com/114385414/192876421-7f4b0d04-1eef-4e33-8762-f1963829b21e.png)

## Задание 2
### В разделе «ход работы» пошагово выполнить каждый пункт с описанием и примером реализации задачи по теме лабораторной работы.
Ход работы:

-Произвести подготовку данных для работы с алгоритмом линейной регрессии. 10 видов данных были установлены случайным образом, и данные находились в линейной зависимости. Данные преобразуются в формат массива, чтобы их можно было вычислить напрямую при использовании умножения и сложения.
```py
In [ ]:
#Import the required modules, numpy for calculation, and Matplotlib for drawing
import numpy as np
import matplotlib.pyplot as plt
#This code is for jupyter Notebook only
%matplotlib inline

# define data, and change list to array
x = [3,21,22,34,54,34,55,67,89,99]
x = np.array(x)
y = [2,22,24,65,79,82,55,130,150,199]
y = np.array(y)

#Show the effect of a scatter plot
plt.scatter(x,y)
```
Мне вывело несколько точек:

![image](https://user-images.githubusercontent.com/114385414/192879487-ebd9aa6a-84a6-44bd-952b-a82f2a92e491.png)


-Определите связанные функции. Функция модели: определяет модель линейной регрессии wx+b. Функция потерь: функция потерь среднеквадратичной ошибки. Функция оптимизации: метод градиентного спуска для нахождения частных производных w и b.
```py
In [ ]:
#The basic linear regression model is wx+ b, and since this is a two-dimensional space, the model is ax+ b

def model(a, b, x):
    return a*x + b

#Tahe most commonly used loss function of linear regression model is the loss function of mean variance difference
def loss_function(a, b, x, y):
    num = len(x)
    prediction=model(a,b,x)
    return (0.5/num) * (np.square(prediction-y)).sum()

#The optimization function mainly USES partial derivatives to update two parameters a and b
def optimize(a,b,x,y):
    num = len(x)
    prediction = model(a,b,x)
    #Update the values of A and B by finding the partial derivatives of the loss function on a and b
    da = (1.0/num) * ((prediction -y)*x).sum()
    db = (1.0/num) * ((prediction -y).sum())
    a = a - Lr*da
    b = b - Lr*db
    return a, b

#iterated function, return a and b
def iterate(a,b,x,y,times):
    for i in range(times):
        a,b = optimize(a,b,x,y)
    return a,b
```
-Начать итерацию
Шаг 1 Инициализация и модель итеративной оптимизации
```py
In [ ]:
#Initialize parameters and display
a = np.random.rand(1)
print(a)
b = np.random.rand(1)
print(b)
Lr = 0.000001

#For the first iteration, the parameter values, losses, and visualization after the iteration are displayed
a,b = iterate(a,b,x,y,1)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
Появилась линия(оно живое):

![image](https://user-images.githubusercontent.com/114385414/192880217-492c7faf-02b0-4309-ae69-dcdc702b72e7.png)

Шаг 2 На второй итерации отображаются значения параметров, значения потерь и эффекты визуализации после итерации
```py
In [ ]:
a,b = iterate(a,b,x,y,2)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
Всё также..

![image](https://user-images.githubusercontent.com/114385414/192880482-dcbe3bbb-7a9c-4eda-8a15-671a6de29762.png)

Шаг 3 Третья итерация показывает значения параметров, значения потерь и визуализацию после итерации
```py
In [ ]:
a,b = iterate(a,b,x,y,3)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
Жду изменений..

Шаг 4 На четвертой итерации отображаются значения параметров, значения потерь и эффекты визуализации
```py
In [ ]:
a,b = iterate(a,b,x,y,4)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
Я точно всё делаю верно?

![image](https://user-images.githubusercontent.com/114385414/192880860-0f108818-07ad-43f6-afe5-daa01ea0551b.png)

Шаг 5 Пятая итерация показывает значение параметра, значение потерь и эффект визуализации после итерации
```py
In [ ]:
a,b = iterate(a,b,x,y,5)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
...

Шаг 6 10000-я итерация, показывающая значения параметров, потери и визуализацию после итерации
```py
In [ ]:
a,b = iterate(a,b,x,y,10000)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
УРА! ПОБЕДА!
Новая линия, она точно стала больше похожа на ту, что была построена по средним значениям

![image](https://user-images.githubusercontent.com/114385414/192881181-5e16a899-7fec-484d-99a4-f346d7302fc8.png)

Этот опыт довольно наглядно показывает, что при малом колличестве итераций, разницы нет, но уже на 1000-че результат более чем заметен

## Задание 3
### Должна ли величина loss стремиться к нулю при изменении исходных данных? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ.

- Я изменю начальные точки 

![image](https://user-images.githubusercontent.com/114385414/192883477-1b805c6b-1adb-4626-8e33-646ccd30dbfc.png)

- Произвожу одну итерацию

![image](https://user-images.githubusercontent.com/114385414/192883961-1396362d-df7a-4c3a-9a38-522aa16b78ce.png)

Значение loss не ноль

- Теперь ещё 999

![image](https://user-images.githubusercontent.com/114385414/192884108-1831b297-2c74-4b49-b513-bac89758e761.png)

До сих пор не ноль

- Ещё 9000

![image](https://user-images.githubusercontent.com/114385414/192884264-b39af882-8c78-4763-8e43-5bfd43351180.png)

Да, точно к нулю не уходит

## Задание 4
### Какова роль параметра Lr? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ. В качестве эксперимента можете изменить значение параметра.

Уверенно тыкаемся:
- Что если ноль?

![image](https://user-images.githubusercontent.com/114385414/192884812-20290db4-7683-46b5-984f-b503d4963072.png)

![image](https://user-images.githubusercontent.com/114385414/192884958-ed47931b-5d12-42f8-bfff-ed0fe6191234.png)

![image](https://user-images.githubusercontent.com/114385414/192885078-a167d2e1-15b2-456d-845b-83d5179026a2.png)

Линия просто падает

- Что если один?

![image](https://user-images.githubusercontent.com/114385414/192885307-49fabd30-3056-48bc-a5db-fad6a0d38a71.png)

Улетела в космос

- Теперь смотрим в код и делаем вывод, что lr - кэф для исменения положения линии

![image](https://user-images.githubusercontent.com/114385414/192885606-22feafb3-f566-44f7-b3ea-e7784b8a25c3.png)

-Проверяем на 100 итерациях изменяя кэф в 10 раз каждый запуск
0.1 - код не работает

![image](https://user-images.githubusercontent.com/114385414/192886082-0e31e106-4742-4c0c-a646-0bb1e2bd2019.png)

0.01 - так, это плохо

![image](https://user-images.githubusercontent.com/114385414/192886231-dc168fd1-d577-43d9-9d25-94f4c28c90f8.png)

0.001 - всё также

0.0001 - оно работает

![image](https://user-images.githubusercontent.com/114385414/192886403-e1ec9f0e-c418-4b60-b138-60d7703ad11d.png)

0.00001 - не вижу разницы

![image](https://user-images.githubusercontent.com/114385414/192886564-2dfd8067-16ff-4393-82b9-1488fcccda23.png)

0.0000001 - палка явно стала ниже

![image](https://user-images.githubusercontent.com/114385414/192886632-14341551-b0ff-40bb-81c8-8d5da339519c.png)

0.0000001 - при 10000 итераций снова вернулась

![image](https://user-images.githubusercontent.com/114385414/192886738-23b5a143-e166-4726-b525-30b7e5a2a627.png)

Уменьшим колличество повторений на меньших значениях

![image](https://user-images.githubusercontent.com/114385414/192887290-a4dfec44-818a-47d5-b6e2-3c11396e3ad2.png)

![image](https://user-images.githubusercontent.com/114385414/192887391-17e758c4-605d-4fb4-a305-d1dfe5423d5c.png)

Вопрос закрыт, догадка успешна

LR влияет на резкость изменения положения линии за итерацию

Если мы хотим сделать меньше итераций - мы его увеличиваем

Хотим повысить точность - уменьшаем

## Выводы
Очень давно не трогал Python, но я рад, что смог разобраться (на первый взгляд выглядело очень страшно)
Первый раз в жизни использовал jupyter, мне очень понравилось, так я ещё с кодом не работал

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
