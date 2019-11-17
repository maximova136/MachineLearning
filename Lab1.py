#!/usr/bin/env python
# coding: utf-8

# # Lab 1 by Maximova Viktoria

# # Задача
# #### Классифицировать крафтовые напитки, описанные набором признаков, по их стилю.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# - `abv` - крепость напитка, где 0 - безалкогольный напиток, а 1 - чистый спирт
# - `ibu` - уровень горечи напитка (по таблице international bittering units)
# - `id` - уникальный ID
# - `name` - название пива
# - `style` - стиль пива (лагер, эль, IPA, и т.д.)
# - `brewery_id` - уникальный ID пивоварни, которая произвела напиток
# - `ounces` - размер банки пива (в унциях)
# 
# 
# - `name` - название пивоварни
# - `city` - город, в котором расположена пивоварня
# - `state` - штат, в котором расположена пивоварня

# In[2]:


# url = "https://www.kaggle.com/nickhould/craft-cans/"
# get beers.csv and breweries.csv
data = pd.read_csv('beers.csv', sep=',', header=0, engine='python')


# In[3]:


type(data)


# In[4]:


data


# Видно, что в `data` есть лишние столбцы - первый (так как дублирует индексацию) и четвертый (уникальный индекс не является значащим параметром). Уберем лишние столбы:

# In[5]:


data = data.drop(columns="Unnamed: 0")
data = data.drop(columns="id")


# In[6]:


data.tail()


# In[7]:


data2 = pd.read_csv('breweries.csv', sep=';', header=0, engine='python')


# In[8]:


data2


# В описании данных было заявлено, что breweries.csv содержит 4 поля, но их оказалось больше. Уберем лишние (начиная с 5-го).

# In[9]:


data2.dtypes


# In[10]:


data2 = data2.drop(columns="Unnamed: 4")
data2 = data2.drop(columns="Unnamed: 5")
data2 = data2.drop(columns="Unique states ")
data2 = data2.drop(columns="Unnamed: 7")
data2 = data2.drop(columns="Count ")


# In[11]:


data2.dtypes


# In[12]:


data2.rename(columns={'Unnamed: 0':'brewery_id'}, inplace=True)


# In[13]:


data2.tail()


# Пришло время объединить `data` и `data2`. Добавим в `data` три новых поля - name_brewery, city, state и заполним их значениями из `data2`

# In[14]:


data.head()


# In[15]:


data2.head()


# In[16]:


data2.dtypes


# In[17]:


data['brewery_name']=''
data['city']=''
data['state']=''


# In[18]:


print(data.head())


# In[19]:


data.dtypes


# In[20]:


import copy
d = copy.deepcopy(data)


# Заполним новые поля данными из `data2`

# In[21]:


for i, row in d.iterrows():
    for j, row2 in data2.iterrows():
        if row2['brewery_id'] == row['brewery_id']:
            d['brewery_name'][i] = row2['name']
            d['city'][i] = row2['city']
            d['state'][i] = row2['state']
            break


# In[22]:


d


# In[23]:


# Теперь нам не нужен столбец `brewery_id`
d = d.drop(columns="brewery_id")


# In[24]:


d[['style', 'state']]


# In[25]:


d.describe()


# In[26]:


# Посмотрим, как данные коррелируют друг с другом
from pandas.plotting import scatter_matrix
scatter_matrix(d, alpha = .01, figsize = (10, 10))
pass


# In[27]:


d.corr()


# Числовые признаки не сильно коррелируют между собой.

# In[28]:


plt.plot(data['abv'], data['ibu'], 'o', alpha = 0.05)
plt.xlabel('abv')
plt.ylabel('ibu')
pass


# In[29]:


d['style'] == 'American IPA'


# In[30]:


plt.figure(figsize = (10, 10))

plt.scatter(d[d['style'] == 'American IPA']['abv'],
            d[d['style'] == 'American IPA']['ibu'],
            alpha = 0.15,
            label = 'IPA',
            color = 'b')

plt.scatter(d[d['style'] == 'American Pale Ale (APA)']['abv'],
            d[d['style'] == 'American Pale Ale (APA)']['ibu'],
            alpha = 0.05,
            label = 'APA',
            color = 'r')

plt.xlabel('abv')
plt.ylabel('ibu')
plt.legend()
plt.grid()


# In[31]:


d['style'].unique()


# ## Подготовка данных

# In[32]:


categorical_columns = [c for c in d.columns if d[c].dtype.name == 'object']
numerical_columns   = [c for c in d.columns if d[c].dtype.name != 'object']
print(categorical_columns)
print(numerical_columns)


# In[33]:


for c in categorical_columns:
    print(c, d[c].unique())


# ### Обработка пропусков

# In[34]:


print(d[categorical_columns].count())
print(d[numerical_columns].count())


# In[35]:


d.index


# Мы имеем пропущенные значения крепости `abv` - таких объектов не очень много, но все-таки не хотелось бы их терять. Заполним пропущенные значения медианными.
# 
# Для поля `ibu` пропущенные значения следует заменить 0, так как в напитке вполне может отсутствовать характеристика горечи.
# 
# Есть 5 объектов с незаполненным полем `style`- эти объекты можно убрать.

# In[36]:


d.dropna(subset=['style'], inplace=True)


# In[37]:


d.index


# In[38]:


print(d[categorical_columns].count())
print(d[numerical_columns].count())


# Заполним `ibu`

# In[39]:


d['ibu'] = d['ibu'].fillna(0)


# In[40]:


d['ibu']


# In[41]:


m = d['abv'].median()
print(m)


# In[42]:


print(d['abv'].max())
print(d['abv'].min())


# In[43]:


d['abv'].unique()


# In[44]:


d['abv'].fillna(m).unique()


# In[45]:


d['abv'] = d['abv'].fillna(m)


# In[46]:


d.index


# In[47]:


print(d[categorical_columns].count())
print(d[numerical_columns].count())


# In[48]:


# Сохраним полученную таблицу в файл
export_csv = d.to_csv('data.csv')


# Пропущенные значения обработаны.

# In[49]:


d = pd.read_csv('data.csv', sep=',', header=0, engine='python')
d = d.drop(columns="Unnamed: 0")


# ### Векторизация

# In[50]:


data_describe = d.describe(include = [object])
binary_columns    = [c for c in categorical_columns if data_describe[c]['unique'] == 2]
nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]
print(binary_columns, nonbinary_columns)


# In[51]:


data_describe


# В наших данных отсутствуют бинарные категориальные признаки. Более того, для некоторых признаков уникальных значений очень много. Для признака `name` почти каждое значение - уникально. Удалим этот столбец позже, с остальными признаками сделаем векторизацию

# In[52]:


print(d['style'].unique())


# In[53]:


class_columns = [c for c in categorical_columns if data_describe[c]['unique'] == 99]
left_columns  = [c for c in categorical_columns if data_describe[c]['unique'] != 99]
left_columns.remove('name')
print(class_columns, left_columns)


# In[54]:


data_class = pd.get_dummies(d[class_columns])
print(data_class.columns)
print(len(data_class.columns))

data_cat = pd.get_dummies(d[left_columns])
print(data_cat.columns)
print(len(data_cat.columns))


# ### Нормализация количественных признаков

# In[55]:


data_numerical = d[numerical_columns]
data_numerical.describe()


# In[56]:


data_numerical = (data_numerical - data_numerical.mean(axis = 0))/data_numerical.std(axis = 0)


# In[57]:


data_numerical.describe()


# ### Соединяем всё в одну таблицу

# In[58]:


print(d.shape)
d.head()


# In[59]:


d2 = pd.concat((data_numerical, data_class, data_cat), axis=1)
print(d2.shape)
print(d2)


# In[60]:


d2.describe()


# ## Обучающая и тестовая выборка

# In[61]:


X = d2.drop(data_class.columns, axis='columns')


# In[62]:


y = d2[data_class.columns]


# In[63]:


X.head()


# In[64]:


y.head()


# In[65]:


feature_names = X.columns


# In[66]:


feature_names


# In[67]:


# X = X.to_numpy()


# In[68]:


# y = y.to_numpy()


# In[69]:


print(X.shape)
print(y.shape)


# In[70]:


print(type(X))
print(type(y))


# In[71]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

N_train, _ = X_train.shape 
N_test,  _ = X_test.shape 

print(N_train, N_test)


# ## $k$NN

# In[72]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train, y_train)


# In[73]:


y_train_predict = knn.predict(X_train)
y_test_predict = knn.predict(X_test)

err_train = np.mean(y_train != y_train_predict)
err_test  = np.mean(y_test  != y_test_predict)


# In[74]:


print(err_train)


# In[75]:


print(err_test)


# In[76]:


print("Mean error:")
print(np.mean(err_train), np.mean(err_test))
print("Maximum error:")
print(np.max(err_train), np.max(err_test))
print("Median:")
print(np.median(err_train), np.median(err_test))


# In[77]:


plt.hist(err_train)

plt.hist(err_test)
# In[78]:


from sklearn.metrics import confusion_matrix

# print(confusion_matrix(y_test, y_test_predict))
# к сожалению, матрицу ошибок нельзя посчитать для многомерной классификации:
# --> 255         raise ValueError("%s is not supported" % y_type)
#     256 
#     257     if labels is None:

# ValueError: multilabel-indicator is not supported


# Получаем, что в ошибка среднем для каждого класса - 0,93% на обучающей выборке и 0,96% на тестовой. Но нас интересует максимальная ошибка, тогда получаем 12,4% на обучающей выборке и 13,2% на тестовой.
# Попробуем уменьшить тестовую ошибку, варьируя параметр `n_neighbours`.

# ### Подбор параметров

# In[79]:


from sklearn.model_selection import GridSearchCV
nnb = [1, 3, 5, 10, 15, 20, 25, 35, 45, 55]
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid = {'n_neighbors': nnb}, cv=10)
grid.fit(X_train, y_train)


# In[80]:


best_cv_err = 1 - grid.best_score_
best_n_neighbors = grid.best_estimator_.n_neighbors
print(best_cv_err, best_n_neighbors)


# In[81]:


knn = KNeighborsClassifier(n_neighbors = best_n_neighbors).fit(X_train, y_train)

err_train_best = np.mean(y_train != knn.predict(X_train))
err_test_best  = np.mean(y_test  != knn.predict(X_test))


# In[82]:


print(err_train_best, err_test_best)


# In[83]:


print("Mean error:")
print(np.mean(err_train_best), np.mean(err_test_best))
print("Maximum error:")
print(np.max(err_train_best), np.max(err_test_best))
print("Median:")
print(np.median(err_train_best), np.median(err_test_best))


# In[84]:


plt.hist(err_train_best)


# In[85]:


plt.hist(err_test_best)


# In[86]:


plt.style.use('seaborn-deep')
plt.hist([err_train, err_train_best], label=['first', 'best'])
plt.legend(loc='upper right')


# In[87]:


plt.style.use('seaborn-deep')
plt.hist([err_test, err_test_best], label=['first', 'best'])
plt.legend(loc='upper right')


# По графикам и данным видно, что на обучающей выборке параметр `n_neighbors=1` улучшил результаты предсказаний до средней ошибки в 0,08% и максимальной ошибки в 0,7%. К сожалению, модель переобучилась, так что на тестовой выборке мы получаем намного более худшие результаты, чем на обучающей выборке - 1,4% средняя ошибка и 15,2% - максимальная.
# Для этого метода результат получился лучше при первоначальном количестве ближайших соседей, равном 10.

# ## Random Forest

# In[88]:


from sklearn import ensemble
rf = ensemble.RandomForestClassifier(n_estimators = 1000)
rf.fit(X_train, y_train)

err_train_rf = np.mean(y_train != rf.predict(X_train))
err_test_rf  = np.mean(y_test  != rf.predict(X_test))


# In[89]:


print(err_train_rf, err_test_rf)


# In[90]:


print("Mean error:")
print(np.mean(err_train_rf), np.mean(err_test_rf))
print("Maximum error:")
print(np.max(err_train_rf), np.max(err_test_rf))
print("Median:")
print(np.median(err_train_rf), np.median(err_test_rf))


# In[91]:


plt.style.use('seaborn-deep')
plt.hist([err_train, err_train_rf], label=['first', 'random-forest'])
plt.legend(loc='upper right')


# In[92]:


plt.style.use('seaborn-deep')
plt.hist([err_test, err_test_rf], label=['first', 'random-forest'])
plt.legend(loc='upper right')


# Итак, на тестовой выборке получаем среднюю ошибку 0,9% и максимальную 10%, что лучше первого запуска метода ближайших соседей, где мы получили 0,96% и 13,2%. Этим запуском мы получили оптимальное решение.

# ### Значимость признаков

# In[93]:


np.sum(rf.feature_importances_)


# In[94]:


importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

N, d = X.shape
print("Feature ranking:")
for f in range(d):
    print("%2d. feature '%5s' (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))


# In[95]:


d_first = 20
plt.title("Feature importances")
plt.bar(range(d_first), importances[indices[:d_first]], align = 'center', color = 'r')
plt.xticks(range(d_first), feature_names[indices[:d_first]], rotation = 90)
plt.xlim([-1, d_first])


# ## Вывод

# Таким образом, минимальная ошибка алгоритмов на тестовой выборке составила 10%. Оптимальные результаты продемонстрировал алгоритм случайного леса, не сильно опередив метод ближайших соседей с k=10. 
