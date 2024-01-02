# -*- coding: utf-8 -*-
"""
Intelligence artificielle en Python
Premier réseau de neurones pour apprendre à multiplier par 3
"""

from keras import *
import time
model=Sequential()

# La couche d'entrée, units doit être égale à 1
# Dense signifie que chaque neurone de cette couche et lié à chaquue neurone de la couche suivante
# Units=3 signifie que l'on créé 3 neurones
print("Ajout de la couche d\'entrée")
model.add(layers.Dense(units=3,input_shape=[1]))

# La couche intermédiaire
print("Ajout de la couche intermédiaire")
model.add(layers.Dense(units=5))

# La dimension de la couche de sortie doit être égale à 1, on attend un résultat en chiffres
print("Ajout de la couche de sortie")
model.add(layers.Dense(units=1))

# Données d'entrée
entree=[1,2,3,4,5]

# Données de sortie attendues
sortie=[3,6,9,12,15]

# Création du réseau de neurones (optimiseur adam ou sgd)
print("Création du réseau de neurones")
model.compile(loss='mean_squared_error',optimizer='sgd')

time.sleep(3)

# Entrainement du réseau de neurones sur 2000 passages
print("Entrainement du réseau de neurones")
model.fit(x=entree,y=sortie,epochs=2000)

# Prédire les nombres
result=str(round(model.predict([6])[0][0]))
print("Résultat de 6 multiplié par 3 : " + result)

result=str(round(model.predict([33])[0][0]))
print("Résultat de 33 multiplié par 3 : " + result)

result=str(round(model.predict([12])[0][0]))
print("Résultat de 12 multiplié par 3 : " + result)

result=str(round(model.predict([1000])[0][0]))
print("Résultat de 1000 multiplié par 3 : " + result)




