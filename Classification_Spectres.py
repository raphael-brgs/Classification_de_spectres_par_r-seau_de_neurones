import numpy as np
from math import exp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter 

## les sons doivent être recuperées à l'aide du fichier txt produit par la
## le spectre de Audacity

## Fonction pour transformer le fichier .txt en numpy array Frequence, Niveau
## Le fichier .txt doit être dans le repertoir courant, les virgule doivent etre en point pour le type float  

def txt_to_array(file):
    spectre = open (file)
    ligne1 = spectre.readline() ## Ne pas traiter la premiere ligne qui ne contient pas de val
    lignes = spectre.readlines() ## recuperation des lignes [ chaque ligne est comme suit " Freq Niveau " ]
    freq = []
    niveau = []

    for l in lignes:
        lsplit = l.split() # Separation de la ligne en deux case de liste [ 0 => Freq ; 1 => Niveau ]
        freq.append(float(lsplit[0])) # list de freq 
        niveau.append(float(lsplit[1])) # liste de niveau

    a = [freq,niveau] #crée un array Freq , niveau 

    spectre.close()
    return a

## fonction pour decalé les niveau en dB initalement negative ###
## on recherche le niveau min du tableau, et on ajoute la valeur absolue de ce niveau ##

def ajuster(array):
    minimum = max(array[1]) + 1.0
    for i in range(len(array[1])):
        if array[1][i] != 0.0 and array[1][i] < minimum :
            minimum = array[1][i] 
    for i in range(len(array[1])):
        if array[1][i] !=0:
            array[1][i] -= minimum
    return array

### fonction pour exponentié l'array de sorte à faire apparaitre les niveau significatif

def exponentier(array):
    for i in range(len(array[1])):
        array[1][i] = exp(array[1][i])
    return array
## algorithme pour que les données soit reduit à la même echelle
def normaliser_Moy(array):
    Moy = 0.0
    for i in range(len(array[1])):
        Moy += array[1][i] / len(array[1])
    for i in range(len(array[1])):
        array[1][i] = (array[1][i] / Moy) * 100 
    return array
    
def normaliser_Max(array):
    maximum = max(array[1])
    for i in range(len(array[1])):
        if maximum != 0:
            array[1][i] = (array[1][i] / maximum)*100
    return array

def redresser(array): ## algorithme pour annuler l'effet de decroissance générale des niveau
    new_array = [array[1][0]]
    delta = 0
    card = len(array[1])
    for i in range(card-1):
        if array[1][i]*array[1][i+1] == 0.0:
            card -= 1
        else:
            delta += (array[1][i]-array[1][i+1]) / (card - 1)
    for k in range(1,len(array[1])):
        if array[1][k]*array[1][k-1] != 0.0:
            new_array.append((array[1][k-1]) + delta*k )
        else:
            new_array.append(array[1][k])
    return [array[0],new_array]
         

def filtre_passe_haut(array,wc): ## wc est la frequence de coupure du filtre
    k = 0
    nb = 0
    while array[0][k] < wc:
        nb += 1
        k += 1
    for i in range(len(array[1])-(len(array[1])-nb)):
        array[1][i] = 0.0
    return array

def filtre_passe_bas(array,wc): ## wc est la frequence de coupure du filtre
    for i in range(len(array[1])):
        if array[0][i] > wc:
            array[1][i] = 0.0
    return array

def filtre_passe_bande(array,wc1,wc2):
    return filtre_passe_bas(filtre_passe_haut(array,wc1),wc2)

def txt_to_array25(file):
    a = ajuster(filtre_passe_bas(redresser(ajuster(txt_to_array(file))),6000.0))
    return normaliser_Max(savgol_filter(a,5,3))
        
                 

####### début du réseau de neuronne #############

##### Données pour l'apprentissage , le spectre est une array de 2,511 

## tableau o :
o_data =  txt_to_array25("spectre_o_data_seul_raphael.txt")
o =  txt_to_array25("spectre_o_seul_raphael.txt")
o_test = txt_to_array25("spectre_o_test_seul_raphael.txt")




##tableau on  :
on_data = txt_to_array25("spectre_on_data_seul_raphael.txt")
on = txt_to_array25("spectre_on_seul_raphael.txt")
on_test = txt_to_array25("spectre_on_test_seul_raphael.txt")


##dictionnaire des données : 0 sont les o et 1 les on

d = { 0 : [o_data] , 1 : [on_data] }

##### Création du réseau ############ type MLP 

### Surcouche pour crée une matrice poids au hasard ##
from numpy import random
def random_within(a, b):
    return (b-a) * random.random() + a
def random_list(n, a, b):
    return [random_within(a, b) for i in range(n)]
def init_poids(dimEntree, dimSortie): 
    poids = [random_list(dimSortie, -25, 25) for i in range(dimEntree)]
    return poids


### La matrice poids se définie : Dim entrée => poids de chaque Niveau [511]
###                               Dim sortie => nombre de sons à reconnaitre [2]


##### Fonction de combinaison et fonction d'activation ##############
import math

## On chosis un reseau de type MLP (multi-layer perceptron), la fonction de combinaison
## est le produit scalaire entre les vecteurs d'entrées et les vecteurs de poids synaptiques 

def calcul_neurone(j, e, poids):
## j est le numero du son à reconnaitre , e est l'array du son
    resultat = 0.0
    count = 0 
    for i in range (len(e[1])): 
        resultat = resultat + e[1][i] * poids[i][j] 
    if resultat > 0:
        boolean = True
    else:
        boolean = False 
    seuil = int(boolean) ## Correspond a la fonction d'activation [seuil 0 1]
    return seuil

def calcul_reseau(e , poids ): ##Cette fonction va passer la fonction de combinaison aux neronnes des deux sons 
    resultats = [calcul_neurone (i, e, poids) for i in range(2)]
    return resultats

##### Apprentissage du reseau ######
## On va forcer la sortie d'un reseau en fonction de l'entrée et de la sortie attendue

def apprendre_neurone(e, poids, j, sortie_attendue):
    h = 10 ## ceci est le parametre d'apprentissage, a modifier en cas de divergence de l'apprentissage
    valeur_calculee = calcul_neurone ( j, e, poids)
    for i in range (len(poids[0])) : 
        valeur_desiree = 0
        if sortie_attendue == j:
            valeur_desiree = 1
        poids[i][j] = poids[i][j] + (valeur_desiree - valeur_calculee) * e[1][i] * h
    return poids

def apprendre_reseau(e, poids, sortie_attendue):
    for j in range(2):
        poids = apprendre_neurone(e, poids, j, sortie_attendue)
    return poids

def apprendre(d, poids):
    for k in d:
        for e in d[k]:
            poids = apprendre_reseau(e, poids, k)
    return poids

##### Test du réseau #########


## on doit mettre un son en entrée comme décrit en haut
son = txt_to_array25("spectre.txt")

     



def performance(son,attendu,poids,rounds):
    echec = 0
    reussite = 0
    indetermine = 0
    for i in range(rounds):
        poids = init_poids( 511 , 2 )

        nb_rounds = 100 ##nombre d'occurences d'apprentissage

        for i in range(nb_rounds):
            poids = apprendre(d, poids)
        resultat = calcul_reseau(son, poids) # on passe le son dans la matrice poids  
        if resultat == attendu:              # on compte le nombre de reussite, d'echec et d'indetermination
            reussite += 1
        elif resultat == [1, 1] or resultat == [0,0]:
            indetermine +=1
        else:
            echec += 1
    a = "reussites: {} , echecs: {}, indetermines: {}".format(reussite,echec,indetermine)
    print (a) 


poids = 0
performance(son,[1, 0],poids,30)


## Affichage de courbes ###
x = np.array(o[0])
y = np.array(o_test[1])
y2 = np.array(on_data[1])

plt.plot(x , y)
plt.plot(x , y2)
#plt.show()




















    


