#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Intelligence artificielle en Python
Second réseau de neurones pour estimer un salaire en fonction de plusieurs critères
"""

from numpy import array
from keras import *
import time

# Creation d'une liste avec les données
# Age / Durée de la formation en années / Expérience professionnelle en années / Sexe 0=Homme 1=Femme / Salaire mensuel en €
data=[
      [18, 2, 0, 0, 2100],
      [32, 4, 10, 0, 3400],
      [44, 2, 20, 0, 3200],
      [60, 6, 34, 0, 5500],
      [20, 2, 2, 1, 2300],
      [24, 4, 0, 1, 2700],
      [44, 5, 21, 1, 3200],
      [52, 2, 32, 1, 3200]
     ]

# Conversion de la liste en tableau
# Le tableau a des index pour les lignes et les colonnes
# Les réseaux neuronaux fonctionnent avec des tableaux
data_array=array(data)

# On définir les données d'entrées et les données de sortie
entree=data_array[0:8,0:4] # Toutes les lignes (index 0-7) colonnes 0-3
sortie=data_array[0:8,4]   # La colonne 4 de toutes les lignes

# Instanciation du modèle / réseau de neuronnes
model=Sequential()

# La couche d'entrée, input_shape doit être égale à 4 par rapport aux 4 colonnes des entrées
# Dense signifie que chaque neurone de cette couche et lié à chaquue neurone de la couche suivante
# Units=16 signifie que l'on créé 16 neurones
print("Ajout de la couche d\'entrée")
model.add(layers.Dense(units=16,input_shape=[4]))

# La couche intermédiaire
print("Ajout deS 2 couches intermédiaires")
model.add(layers.Dense(units=64))
model.add(layers.Dense(units=64))

# La dimension de la couche de sortie doit être égale à 1, on attend un résultat de salaire en chiffres
print("Ajout de la couche de sortie")
model.add(layers.Dense(units=1))

# Création du réseau de neurones (optimiseur adam ou sgd)
print("Création du réseau de neurones")
model.compile(loss='mean_squared_error',optimizer='adam')

time.sleep(3)

# Entrainement du réseau de neurones sur 5000 passages
print("Entrainement du réseau de neurones")
model.fit(x=entree,y=sortie,epochs=5000)

# Prédictions
print("Estimation du salaire d\'un homme de 18 ans avec 1 an de formation, et 1 an d\'expérience professionnelle")
Gabriel=array([[18,1,1,0]])
salaire = str(round(model.predict(Gabriel)[0][0]))
print(salaire + "€")

print("Estimation du salaire d\'une femme de 52 ans avec 7 ans de formation, et 20 ans d\'expérience professionnelle")
Louise=array([[52,7,20,1]])
salaire = str(round(model.predict(Louise)[0][0]))
print(salaire + "€")

print("Comparaison du salaire entre un homme et une femme de 30 ans avec 3 ans de formation, et 7 ans d\'expérience professionnelle")
Nicolas=array([[30,3,7,0]])
Jennifer=array([[30,3,7,1]])
salaireH = str(round(model.predict(Nicolas)[0][0]))
salaireF = str(round(model.predict(Jennifer)[0][0]))
print("Homme : " + salaireH + "€ / Femme : " + salaireF + "€")




