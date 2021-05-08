# Résumé des travaux effectués sur la nasalisation ( Rapport complet disponible dans le dépot )
Projet scientifique
24 Mai 2020
Raphaël BOURGEOIS
Clément CHIDEKH

## Introduction

Au niveau de l’articulation, la nasalisation correspond à la production d’un son lorsque le voile du palais

est abaissé par le relâchement du levator palatini, principal muscle responsable de l’élévation du voile

du palais, de telle sorte que l'air puisse s'échapper par le nez durant la production du son. Il se produit

alors une résonance nasale, qui auditivement se traduit par la production d’un son [n] pour les voyelles

nasales. Par exemple, le son « a » devient « an », le « o » devient alors un « on ». On dit que les voyelles

passent du timbre oral au timbre nasal.

Dans l'alphabet phonétique international, la nasalisation est indiquée par la

présence d'un tilde au-dessus du symbole du son à nasaliser. Les voyelles nasales sont présentes à

hauteur de 5 dans la langue française avec [Ã], [Õ], [Ũ], [Ĩ] même si la différenciation entre les deux

dernières se fait de plus en plus menue: par exemple, on différencie très peu les mots “brin” et “brun”

(Nasalisation, s.d.). Aujourd’hui, parmi les 700 langues répertoriées par Ruhlen, 1975, 150 ont des

voyelles nasales (Vaissière, 1995).


## 1. Les caractéristiques du spectre d’une voyelle nasalisée

Dans cette partie, on s'intéressera exclusivement à la voyelle nasale [Õ] qui s’entend ”on” et à son

équivalent vocal [O]. Afin de déterminer les caractéristiques du [Õ] et ses différences avec le [O] ne

nous appuierons sur les enregistrements de ces syllabes dans le contexte de la phrase “Bonjour Monsieur

Tralipo”. 

A l’aide de ces données, on constate que chez les personnes adultes, la nasalisation s’accompagne d’un

déplacement de la fréquence fondamentale dans les aigus de 20 Hz en moyenne alors que chez l’enfant

c’est le contraire qui se produit. Cependant, un schéma général semble apparaître :

alors que pour le [O] la fréquence fondamentale et la 1ère harmonique sont d’intensité pratiquement

égale dans les 3 cas avant une baisse linéaire pour les harmoniques suivantes, cette baisse “en escalier”

s’exprime dès la 1ère harmonique pour le [Õ]. Ce schéma s’explique par la perte d’intensité de la 1ère

harmonique lors de la nasalisation. En dehors de cela, les voyelles nasales et orales suivent les mêmes

inflexions.

### 2.2 L’ajustement de la cavité pharyngale

Dans la langue française, en plus du mécanisme d’abaissement du voile du palais qui crée

l’amortissement de la 1ère harmonique comme vu précédemment, la nasalisation apparaît également

par l’ajustement de la cavité pharyngale. Cela permet une annulation de certaines harmoniques

expliquant le creux observé autour de 3000 Hz. C’est l’ajustement de cette cavité pharyngale qui nous

permet de distinguer les différentes nasalités de la langue française. Les schémas de la figure 4 illustre

la position de cette cavité pour différentes nasalités.

## 3. Reconnaissance d’une voyelle nasale à l’aide d’un réseau de neurones

Nous avons découvert les réseaux neurones et leur utilité dans le domaine de l’intelligence artificielle

grâce à un projet réalisés dans l’UF Initiation à Python en 2ème année MIC à l’INSA de Toulouse. Lors

de ce projet, il nous était demandé de réalisés en plusieurs étapes guidées un réseau de neurones dans

le but « d’apprendre » à une machine à reconnaitre les chiffres de 0 à 9 dessinées dans une matrice à

base de 0 et de 1, chaque chiffre pouvant être dessiné de façon différente. La figure 5 montre comment

le même chiffre peut être dessiner de deux façons différentes.

A la fin du projet, la machine était effectivement capable de reconnaitre les chiffres de 0 à 9. Nous

avons donc eu l’idée de reprendre cette méthode d’apprentissage afin de l’appliquer à la reconnaissance

d’une voyelle nasale par rapport à sa sœur orale.

### 3.1 Généralités sur les réseaux de neurones

Le réseau de neurones artificiels est un système informatique dont le fonctionnement est similaire à

celui des neurones du cerveau humain. C’est une variété de technologie de Deep Learning

(apprentissage profond), une sous-catégorie du Machine Learning (apprentissage automatique)

(Bastien, 2019). Dans la suite, afin d’en savoir en peu plus sur les réseaux de neurones artificiels, nous

exposerons des parties du cours de Marc Parizeau sur les réseaux de neurones (Parizeau, 2004).

Histoire

Le cerveau d’un homme compte en moyenne 100 milliards de neurones qui lui permettent de réaliser

tous les mécanismes essentiels au bon fonctionnement de l’organisme. Ces neurones sont organisés en

réseau et reliés par des synapses. Cette organisation de la prise de décision a inspiré la plupart des

architectures de réseau de neurones artificiels. Les premières recherches sur ces réseaux remontent à la

fin du 19e siècle par des scientifiques comme Hermann von Helmholtz, Ernst Mach et Ivan Pavlov. Ce

ne sont là que des théories générales sans l’interventions de modèle mathématique précis. Il faut attendre

les années 1940 pour que des scientifiques comme Warren McCulloch, Walter Pitts et Donald Hebb

montrent que de tels réseaux peuvent calculer n’importe quelle fonction logique et proposent une théorie

pour l’apprentissage. La première utilisation concrète d’un réseau de neurones remontent à la fin des

années 1950 avec le « perceptron » développé par Frank Rosenblatt. Ce réseau est capable de

reconnaître des formes. Il a été démontré par la suite que des réseaux de la complexité du « perceptron

» ne pouvait résoudre qu’une quantité limitée de problème. Cette découverte a ralenti les recherches sur

le sujet jusqu’aux années 1980 où l’algorithme de rétropropagation des erreurs a été découvert qui

répond au critique faites aux réseaux de neurones. Depuis, de nouvelles théories, structures et

algorithmes sont constamment développés.

**Domaines d’application**

Les réseaux de neurones sont utilisés de nos jours dans de nombreuses applications comme par

exemples des systèmes d’autopilotage d’avion, de guidage pour automobile, de synthèse de la parole,

de prévisions sur les marchés monétaires, le diagnostic médical et dans de nombreux autres domaines.

Le perceptron multicouche

Sans s’aventurer dans le formalisme mathématique du réseau perceptron, définissions ce qu’est ce type

de réseau. Le perceptron multicouche est un réseau à propagation vers l’avant c’est-à-dire que

l’information se propage de l’entrée vers la sortie, sans rétroaction. Il bénéficie d’un apprentissage

supervisé par correction des erreurs, c’est-à-dire que les données d’entrées sont couplées avec leurs

sorties désirées (appelées cibles). Cette méthode d’apprentissage est également appelée l’apprentissage

par exemple. La correction des erreurs se fait à chaque itération de l’apprentissage.

Ce réseau de neurones et l’un des réseaux les plus utilisés pour des problèmes d’approximation, de

classification et de prédiction. Il se compose d’entrées, de plusieurs couches cachées de neurones

matérialisés par des matrices poids et des sorties. Dans la matrice poids, les poids dans la matrice de

chaque neurone sont des scalaires qui sont modifiés lors de la phase d’apprentissage du réseau et qui,

après apprentissage, permet de calculer la sortie en fonction de l’entrée.

### 3.2 Les objectifs de notre algorithme utilisant un réseau de neurones

Nous utiliserons donc le modèle du réseau de neurones perceptron pour faire la différence entre une

voyelle orale et sa variante nasale. Les voyelles que l’on souhaitera différencier sont le ɔ̃ et o (« on » et

« o »). L’entrée de l’algorithme sera le spectre du son de la voyelle prononcée. Ce spectre est récupéré

sous la forme d’un tableau à deux colonnes, représentant les différente niveaux (en dB) en fonction de

la fréquence (en Hz). Ce tableau est récupéré à l’aide du logiciel Audacity (Audacity). La sortie sera

donc la voyelle identifiée par le réseau. Le premier objectif sera que le réseau puisse identifier

correctement un spectre qu’il a déjà dans sa base de données, c’est-à-dire un spectre qu’il a déjà appris

à reconnaître. Ensuite, le réseau devra reconnaître un spectre qu’il ne connaît pas. On se limitera à la

reconnaissance de voyelles issues du même locuteur, les spectres d’un locuteur à l’autre étant trop

différents pour être gérés par le modèle utilisé. Le réseau sera codé et compilé avec IDLE en Python

3.7

### 3.3 Les étapes de réalisation du réseau de neurones

Le code complet et commenté de l’algorithme étant donné en annexe de ce rapport, il s’agira ici de

décrire les grandes lignes de la réalisation du réseau de neurones ainsi que l’objectif des sous-

programmes le composant, sans s’attarder à expliquer ligne par ligne ce que fait le code.

**Structuration des données d’apprentissage**

Les spectres des voyelles en entrée sont récupérés à l’aide de Audacity sous la forme de fichiers textes

(.txt). Il fallait donc convertir les fichiers textes en objet exploitable et modifiable facilement en python.

Nous avons donc choisi de ranger les niveaux et les fréquences dans deux sous-listes d’une même liste.

Ces données peuvent donc être parcourues itérativement à la façon d’un tableau à deux entrées ou d’une

matrice. La lecture du fichier texte et le stockage dans le tableau se fait à l’aide des méthodes .readlines()

et .append(). Après avoir stocké les données dans des tableau, il y a une série de transformation à

effectuer le tableau afin que l’apprentissage se passe dans de bonnes conditions.

Tout d’abord, les spectres enregistrés sont composés de niveaux négatifs en dB. Or, l’apprentissage du

réseau se fera à l’aide d’un produit scalaire entre l’entrée et la matrice poids, tout cela avec un facteur

0, -1 ou 1 en fonction du poids que l’on souhaite donnée à un neurone. Donc si l’entrée comporte que

des valeurs négatives, cela donnera lieu à des faux poids positifs. En effet, le produit de deux nombres

positifs est négatif. De plus, les relations d’ordre des valeurs négatives sont inversées par rapport aux

distances à zéro. Il faut donc travailler avec des valeurs positive afin les grandes valeurs en dB (les pics)

puissent avoir un plus grand impact que les petites valeurs. Cette transformation est faite avec la

fonction ajuster().

Ensuite, lorsque qu’on observe la courbe des spectres avec la méthode pyplot de matplotlib on remarque

une décroissance générale des niveaux. Cette décroissance générale peut être problématique car des

pics qui se situeraient dans des hautes fréquences auront un poids moins important que des pics en

basses fréquences. Il faut donc « redresser » la courbe. Pour cela, nous avons développé une fonction

redresser() qui calcul la décroissance moyenne entre deux valeurs successives sur toute la courbe et qui

ensuite à chaque niveau de rang i lui ajoute le produit cette décroissance moyenne avec le rang. La

figure 6 illustre les changements effectués par la fonction

## Figure 7 - Spectre après tous les traitements réalisés

Nous avons ensuite remarqué que dans les spectres du locuteur des voyelles, les niveaux après 10 kHz

ne sont plus significatifs. Afin qu’ils ne soient pas pris en compte lors de l’apprentissage, nous avons

filtrer les fréquences, ne laissant que celles en dessous de 10 kHz et forçant les autres à 0. Nous avons

donc codé une fonction filtre_passe_bas().

Les spectres ont ensuite été réajuster et passer au filtre de Savitzky-Golay afin de lisser les courbes et

laisser les pics apparents. Ce filtre est le seul filtre que nous n’avons pas codé nous-même, il a été

récupérer sur scipy.signal. Il passe par l’approximation de la courbe par un développement limité à un

ordre variable. Enfin, tous les spectres sont « normalisés » par rapport à leurs maximum respectif afin

d’avoir des niveaux sur la même échelle. Cela néglige les effets des « j’ai parlé plus fort en enregistrant

cette voyelle que celle-là ». Dans la figure 7 , on remarque bien que les niveaux sont sur une échelle de

0 à 100.

Les données pour l’apprentissage sont ensuite rangées dans un dictionnaire.

Création de la matrice poids

La matrice poids va modéliser le poids accordé à chaque neurone lors de l’apprentissage afin de calculer

la sortie. Les tableaux des spectres générés par Audacity contiennent 511 valeurs de niveaux, chacune

associé à une fréquence. La matrice poids aura donc 511 neurones d’entrées et deux sorties afin

d’identifier un « o » ou un « on ». Avant l’apprentissage, le poids de chaque neurone est généré

aléatoirement entre -25 et 25.

**Fonction d’activation des neurones pour la sortie**

La fonction d’activation des neurones est la fonction qui permet de calculer la sortie du réseau en

fonction de l’entrée et de la matrice poids. Cette fonction est modélisée par la fonction calcul_neurone().

Cette fonction réalise le produit scalaire de l’entrée avec la matrice poids puis retourne 0 si ce produit

scalaire est négatif ou 1 sinon. C’est donc une fonction seuil. Ce produit scalaire est réalisé 2 fois pour

chaque sortie du réseau. Il y a donc une valeur booléenne associée à la sortie « o » et un à la sortie « on

». Par exemple, lorsqu’un « o » est identifié, la fonction d’activation devrait retourner une liste de la

forme [1, 0]. Les cas [1, 1] et [0, 0] sont donc des cas d’indéterminations.

**Algorithme d’apprentissage du réseau**

L’apprentissage du réseau se fait avec la fonction apprendre(). Cette fonction modifie le poids des

neurones dans la matrice poids en fonction de la valeur de sortie désirée et la valeur réelle calculé par

calcul_neurone(). La formule exacte est :

poids = poids + (valeur_desiree - valeur_calculee) * e * h

poids est le poids du neurone, e est la donnée et h un coefficient d’apprentissage qui permet d’accélérer

l’apprentissage mais s’il est trop grand il présente des risques de divergence d’un poids qui ne devrait

pas être aussi grand.

On remarque que le terme (valeur_desiree - valeur_calculee) peut prendre trois valeurs :

- 0 si la valeur calculée et la valeur désirée sont identiques, le poids n’est donc pas modifié
- - 1 si la valeur désirée est 0 mais la valeur calculée est 1, le poids est donc diminué afin d’arriver
à un poids négatif. C’est pour cela que les données e doivent être positives
- 1 si la valeur désirée est 1 mais la valeur calculée est 0, le poids est donc augmenté afin d’arriver
à un poids positif.

L’apprentissage doit s’effectuer un certain nombre de fois afin que la matrice poids soit fonctionnelle,

on définit ce nombre d’occurrences avec la variable nb_rounds.

Algorithme de test du réseau

Afin de tester ce réseau, nous avons réalisé une fonction de test qui compte le nombre de réussites,

d’échecs et d’indéterminations à la reconnaissance d’un son sur un certain nombre de calculs de la

sortie. Une nouvelle matrice poids est générée à chaque itération de cette fonction de test car, pour une

même matrice poids et pour une même entrée, la sortie calculée sera toujours la même. Cette fonction

de test s’appelle performance. Elle prend en argument le spectre d’entrée, la sortie attendue, la matrice

poids à modifier et le nombre de tests à effectuer.

### 3.4 Les résultats de la reconnaissance à l’aide de cet algorithme

Dans le dictionnaire d’apprentissage, on place un spectre par sortie, c’est-à-dire un spectre « o » et un

spectre « on ». Ces spectres sont appelés respectivement o_data et on_data. Les spectres extérieurs au

dictionnaire d’apprentissage sont également générés et son appelés o , o_test , on et on_test. Tous les

spectres utilisés viennent d’enregistrements du même locuteur.

Reconna **issance d’enregistrements présents dans le dictionnaire d’apprentissage**

Nous souhaitons tester si le réseau est capable de reconnaître les spectres des voyelles qu’il a déjà appris.

Nous exécutons donc le test de performance sur les spectres o_data et on_data. Nous obtenons les

résultats suivants exposés dans la figure 8.

Nous remarquons donc que le réseau arrive parfaitement à faire la différence entre les spectres présent

dans son dictionnaire d’apprentissage.

**Reconnaissance d’enregistrements extérieurs au dictionnaire d’apprentissage**

Nous allons désormais tester si le réseau est capable de reconnaître des spectres de voyelles qui ne sont

pas présent dans le dictionnaire d’apprentissage. Nous exécutons le test de performance sur les spectres

o , o_test , on et on_test. Nous obtenons les résultats suivants exposés dans la figure 9.

Les résultats sont plutôt concluants. On note quand même les 11 cas d’indéterminations sur o_test, ce

qui nous amène à discuter des limites de l’utilisation de ce réseau de neurones.

### 3.5 Les limites de l’utilisation du réseau de neurones développé

Lors des tests, nous avons remarqué de nombreux cas d’indéterminations sur certains spectres, comme

pour le spectre o_test. Cela traduit les limites d’utilisations de cette méthode pour la reconnaissance de

voyelles nasales.

Les spectres issus d’un même locuteur peuvent présenter quelques variations de la position des pics

significatifs de la voyelle ce qui peut induire la reconnaissance en échec ou en indétermination.

Parallèlement, les spectres varient en fonction de l’âge, du sexe et de l’accent du locuteur ainsi qu’en

fonction d’autres paramètres physiologiques liés à la forme de la cavité pharyngale et du levator

palatini. Cela modifie fortement l’allure des spectres et donc la position des pics significatifs, ce qui

peut engendrer d’un locuteur à un autre des indéterminations et des échecs. En cas de changement de

locuteur, il faudra très probablement changer certains des paramètres des filtres que subissent les

données avant exploitations, comme par exemple modifier la fréquence de coupure du filtre passe-bas

ou encore l’ordre du polynôme du filtre de Savitzky-Golay.

Plus généralement, les réseaux de neurones de type perceptron qui possèdent qu’une seule couche

cachée de neurones comme celui que nous avons développé sont utile pour résoudre des problèmes

assez simples. La reconnaissance de matrices représentant les chiffres de 0 à 9 se fait parfaitement bien

avec ce type de réseaux. Lorsque qu’on regarde l’allure générale des spectres que nous proposons au

réseau, on remarque qu’elles sont à l’œil nu très ressemblantes. Le réseau arrive tout de même sur

plusieurs occurrences à reconnaitre plus souvent la bonne voyelle. Cependant, dès que la voyelle est

prononcée un peu différemment par le locuteur, ou lorsque qu’on change de locuteur, la reconnaissance

devient alors impossible. Pour pallier ces problèmes, il faudrait par exemple complexifier le réseau en

augmentant le nombre de couches de neurones et inclure des algorithmes puissants de correction des

erreurs comme par exemple l’algorithme de rétropropagation du gradient stochastique (Parizeau, 2004).

## Conclusion

Ce projet a été pour nous l’occasion de commencer à nous approprier un phénomène qui nous était

inconnu et pourtant auquel on fait appel quotidiennement. Cela nous a permis d’en apprendre un peu

plus sur notre corps. De plus, les circonstances dans lesquelles une partie du projet a été réalisée nous

ont également appris à travailler ensemble, d’une nouvelle manière, sans pouvoir être en contact direct.

Ce contexte a pu en effet engendrer des difficultés niveau de la communication.

Nous avons abordé ce phénomène en plusieurs étapes. Commençant par la perception de ce qu’est la

nasalisation, à l’oreille et dans au niveau de notre voile du palais. Puis, grâce au logiciel Audacity, nous

avons abordé les caractéristiques d’une voyelle nasale qui permet de percevoir ce son. Cela nous a aider

à comprendre et à illustrer les mécanismes physiologiques qui permettent la nasalisation. Enfin, après

avoir abordé le mécanisme sur ces 3 aspects, le développement de l’algorithme nous a permis de mettre

en pratique les notions algorithmiques acquises depuis la 1ère année, d’utiliser un nouveau langage de

programmation et de mettre un premier pas dans le monde du Machine Learning avec une application

directe de l’utilisation d’un réseau de neurones.

## Bibliographie

Audacity. (s.d.).

Bastien, L. (2019, avril 5). _Réseau de neurones artificiels : qu’est_ - _ce que c’est et à quoi ça sert?_

Récupéré sur Le Big Data: https://www.lebigdata.fr/reseau-de-neurones-artificiels-definition

Nasalisation. (s.d.). Récupéré sur Wikimonde.

Parizeau, M. (2004). Réseaux de neurones GIF-21140 et GIF-64326. Université Laval.

Vaissière, J. ( 1995). Nasalité et phonétique.

Delvaux, V., Metens, T., & Soquet, A. (2002). Propriétés acoustiques et articulatoires des voyelles

nasales du français. XXIVèmes Journées d’étude sur la parole, Nancy, 1, 348-352.
