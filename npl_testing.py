#!/usr/bin/env python
# coding: utf-8

# In[137]:


#!pip install spacy
#!pip install sklearn
#!pip install torch
#!python -m spacy download es_core_news_lg
#!python -m spacy download es_dep_news_trf


# In[1]:


#import spacy module
import spacy
import torch

from sklearn import svm


# In[138]:


#load the pretrained model
nlp = spacy.load("es_core_news_lg")


# In[20]:


doc = nlp("Quisiera que las personas comieran mejor")


# In[135]:


# create a class for all differents objectives
class objetivos:
  pobreza = "FIN DE LA POBREZA"
  hambre = "HAMBRE CERO"

train_x_sample = ["Para 2030, erradicar la pobreza extrema para todas las personas en el mundo, actualmente medida por un ingreso por persona inferior a 1,25 dólares al día","Para 2030, reducir al menos a la mitad la proporción de hombres, mujeres y niños y niñas de todas las edades que viven en la pobreza en todas sus dimensiones con arreglo a las definiciones nacionales","Poner en práctica a nivel nacional sistemas y medidas apropiadas de protección social para todos y, para 2030, lograr una amplia cobertura de los pobres y los más vulnerables","Para 2030, garantizar que todos los hombres y mujeres, en particular los pobres y los más vulnerables, tengan los mismos derechos a los recursos económicos, así como acceso a los servicios básicos, la propiedad y el control de las tierras y otros bienes, la herencia, los recursos naturales, las nuevas tecnologías y los servicios económicos, incluida la microfinanciación","Para 2030, fomentar la resiliencia de los pobres y las personas que se encuentran en situaciones vulnerables y reducir su exposición y vulnerabilidad a los fenómenos extremos relacionados con el clima y a otros desastres económicos, sociales y ambientales","Garantizar una movilización importante de recursos procedentes de diversas fuentes, incluso mediante la mejora de la cooperación para el desarrollo, a fin de proporcionar medios suficientes y previsibles para los países en desarrollo, en particular los países menos adelantados, para poner en práctica programas y políticas encaminados a poner fin a la pobreza en todas sus dimensiones","Crear marcos normativos sólidos en el ámbito nacional, regional e internacional, sobre la base de estrategias de desarrollo en favor de los pobres que tengan en cuenta las cuestiones de género, a fin de apoyar la inversión acelerada en medidas para erradicar la pobreza","Para 2030, poner fin al hambre y asegurar el acceso de todas las personas, en particular los pobres y las personas en situaciones vulnerables, incluidos los lactantes, a una alimentación sana, nutritiva y suficiente durante todo el año","Para 2030, poner fin a todas las formas de malnutrición, incluso logrando, a más tardar en 2025, las metas convenidas internacionalmente sobre el retraso del crecimiento y la emaciación de los niños menores de 5 años, y abordar las necesidades de nutrición de las adolescentes, las mujeres embarazadas y lactantes y las personas de edad","Para 2030, duplicar la productividad agrícola y los ingresos de los productores de alimentos en pequeña escala, en particular las mujeres, los pueblos indígenas, los agricultores familiares, los pastores y los pescadores, entre otras cosas mediante un acceso seguro y equitativo a las tierras, a otros recursos de producción e insumos, conocimientos, servicios financieros, mercados y oportunidades para la generación de valor añadido y empleos no agrícolas","Para 2030, asegurar la sostenibilidad de los sistemas de producción de alimentos y aplicar prácticas agrícolas resilientes que aumenten la productividad y la producción, contribuyan al mantenimiento de los ecosistemas, fortalezcan la capacidad de adaptación al cambio climático, los fenómenos meteorológicos extremos, las sequías, las inundaciones y otros desastres, y mejoren progresivamente la calidad del suelo y la tierra","Para 2020, mantener la diversidad genética de las semillas, las plantas cultivadas y los animales de granja y domesticados y sus especies silvestres conexas, entre otras cosas mediante una buena gestión y diversificación de los bancos de semillas y plantas a nivel nacional, regional e internacional, y promover el acceso a los beneficios que se deriven de la utilización de los recursos genéticos y los conocimientos tradicionales y su distribución justa y equitativa, como se ha convenido internacionalmente","Aumentar las inversiones, incluso mediante una mayor cooperación internacional, en la infraestructura rural, la investigación agrícola y los servicios de extensión, el desarrollo tecnológico y los bancos de genes de plantas y ganado a fin de mejorar la capacidad de producción agrícola en los países en desarrollo, en particular en los países menos adelantados","Corregir y prevenir las restricciones y distorsiones comerciales en los mercados agropecuarios mundiales, entre otras cosas mediante la eliminación paralela de todas las formas de subvenciones a las exportaciones agrícolas y todas las medidas de exportación con efectos equivalentes, de conformidad con el mandato de la Ronda de Doha para el Desarrollo","Adoptar medidas para asegurar el buen funcionamiento de los mercados de productos básicos alimentarios y sus derivados y facilitar el acceso oportuno a información sobre los mercados, en particular sobre las reservas de alimentos, a fin de ayudar a limitar la extrema volatilidad de los precios de los alimentos"]
print(train_x_sample[1])

train_y_sample = [objetivos.pobreza,objetivos.pobreza,objetivos.pobreza,objetivos.pobreza,objetivos.pobreza,objetivos.pobreza,objetivos.pobreza,objetivos.hambre,objetivos.hambre,objetivos.hambre,objetivos.hambre,objetivos.hambre,objetivos.hambre,objetivos.hambre,objetivos.hambre]
print(train_y_sample[1])


# In[139]:


# training model
docs = [nlp(text) for text in train_x_sample]
train_x_trf = [doc.vector for doc in docs]
clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_trf, train_y_sample)


# In[51]:


#test some examples
test = ["hambre"]
docs = [nlp(text) for text in test]
test_vector = [doc.vector for doc in docs]

clf_svm.predict(test_vector)


# ### Modelo mas espesifico a las metas

# In[82]:


import pandas as pd

train_raw = pd.read_csv('E:/Documents/Master/Practicas/Code/vocabulary.csv',header=None)
train_raw.columns = ["Objetivo", "Meta", "Texto", "Unknown"]

train_raw.tail()


# In[83]:


train_raw.dtypes


# In[105]:


metas = train_raw['Meta'].drop_duplicates()
metas.head()


# In[110]:


dicc = {}
for i in metas:
    dicc[i] = 'Meta '+i
    
dicc


# In[112]:


train_x = train_raw['Texto']
print(train_x_sample[1])


# In[121]:


train_y = train_raw['Meta']


# In[116]:


# training model
docs = [nlp(text) for text in train_x]
train_x_vectors = [doc.vector for doc in docs]


# In[117]:


clf_svm = svm.SVC(kernel='linear')


# In[122]:


clf_svm.fit(train_x_vectors, train_y)


# In[129]:


#test some examples
test = ["bajar costos de energia"]
docs = [nlp(text) for text in test]
test_vector = [doc.vector for doc in docs]

clf_svm.predict(test_vector)


# In[ ]:




