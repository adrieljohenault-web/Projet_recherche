# projet_recherche
Mmmmmmh

26/04
pour la suite, je pense qu'il faut passer en date par heure arrondie (v1 prend des meseures a 12h34 - > 13h00). A voir comment gérer ca : soit on fait une moyenne sur les tmeps proche, soit en prend le plus proche (ce qui est à mon avis suffisant). ensuite, tout devrati être ok pour faire la fonction de verif

Je n'ai pas fait de la partie IA parce que après la discussion sur comment aborder le problème, je ne voulais pas partir tête baisser sans être sur d'avoir compris

27/04
J'ai fait une foncton pour vérifier si les dates se suivaient par temps de 10 min 
def dates_verif(val : list):
    # vérifie l'intervale des dates. d'après les premières dates, on peut conjecturer qu'il y a une mesure toutes les 10 min à partir de h+4min+ k*10*min
    date = [line[0] for line in val]
    Lverif = [0 for k in range(len(date)-1)]             # Lverif[i] = True si il y a 10 min entre i et i-1

    #formatage des dates : an, mois, jour, heure, minute
    for k in range(len(date)-1):
        date1 = date[k]
        date2 = date[k+1]

        if date1[3] != date2[3]:
            if date1[4] +10 == date2[4] + 60:
                Lverif[k] = True
            else:
                Lverif[k] = False
                print(date1, date2)
        else:
            if date1[4] +10 == date2[4]:
                    Lverif[k] = True
            else:
                Lverif[k] = False
                print(date1, date2)
                
    return Lverif

J'ai montrer que les dates ne se suivaient pas su tout dans certains cas ( je vaias tout casser ) (pour v1,v2,v3)

Dcp j'ai voulu vérifier pour vin3. Et la (roulement de tambours), il n'y pas de trou ! Les heures s'enchainent sur tout l'interval (YESSSSSSS)

def dates_verif(val : list):
    # vérifie l'intervale des dates. d'après les premières dates, on peut conjecturer qu'il y a une mesure toutes les 10 min à partir de h+4min+ k*10*min
    date = [line[0] for line in val]
    Lverif = [0 for k in range(len(date)-1)]             # Lverif[i] = True si il y a 10 min entre i et i-1

    vrai = 0
    #formatage des dates : an, mois, jour, heure, minute
    for k in range(len(date)-1):
        date1 = date[k]
        date2 = date[k+1]

        if date1[2] != date2[2]:
            if date1[3] +1 == date2[3] + 24:
                Lverif[k] = True
            else:
                Lverif[k] = False
                print(date1, date2)
                vrai +=1
        else:
            if date1[3] +1 == date2[3]:
                    Lverif[k] = True
            else:
                Lverif[k] = False
                print(date1, date2)
                vrai +=1
        print(vrai)
                
    return Lverif

après quelques heures de rien, je crois que je suis parti dans la mauvaise direction

fonction de validation que pour toutes les dates de v1,v2,v3, la fonction trouve bien les plus proches (vérification avec v1,v2,v3 validée)
def verif_345(v,vin3):
    count = 0

    for line in v:
        date1 = line[0]
        rep = get_closest_value(date1,vin3)
        if rep == False:
            print(date1)
            count +=1
    return count 





Pour la prochaine fois : Au lieu de calculer tout les points pour la fonction de tansfert, on la modifie pour calculer que les points qui nous interesse vraiment (ceux de points_and_weights), par ce que je pense que c'est ca qui prend du temps
