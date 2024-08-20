import csv
import random
import time
from math import sqrt

# Charger les données depuis un fichier CSV
def load_data(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Ignorer l'en-tête
        data = [row for row in reader]
    return data

#Conversion des données en nombre
def convert_data(data):
    for row in data:
        row[0] = int(row[0])
        row[1] = int(row[1])  # Age
        row[2] = int(row[2])  # Gender
        row[3] = int(row[3])  # Ethnicity
        row[4] = int(row[4])  # ParentEducation
        row[5] = float(row[5])  # StudyTimeWeekly
        row[6] = int(row[6])  # Absences
        row[7] = int(row[7])  # Tutoring
        row[8] = int(row[8])  # ParentSupport
        row[9] = int(row[9])  # Extracurricular
        row[10] = int(row[10])  # Sports
        row[11] = int(row[11])  # Music
        row[12] = int(row[12])  # Volunteering
        row[13] = float(row[13])  # GradeClass


# Scinder les données en deux parties de manière pseudo-aléatoire: jeu de données d'apprentissage(ensemble d'entrainement) et jeu de données de test
def split_data(data, train_size=0.8):
    random.shuffle(data)
    train_count = int(len(data) * train_size)
    train_data = data[:train_count]
    test_data = data[train_count:]
    return train_data, test_data

# Calculer le centroïde de chaque classe
def calculate_centroids(train_data):
    centroids = {}
    class_counts = {}

    for row in train_data:
        class_label = row[-1]
        if class_label not in centroids:
            centroids[class_label] = [0] * (len(row) - 1)
            class_counts[class_label] = 0
        for i in range(1, len(row) - 1):  
            centroids[class_label][i] += row[i]
        class_counts[class_label] += 1

    for class_label in centroids:
        for i in range(0, len(centroids[class_label])):
            centroids[class_label][i] /= class_counts[class_label]
    
    return centroids

# Calcul de la distance euclidienne
def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return sqrt(distance)

# Prédiction de la classe(des éléments dans le jeu de données de test) basée sur le centroïde
def predict(test_data, centroids):
    predictions = []
    for row in test_data:
        distances = {class_label: euclidean_distance(row[1:-1], centroid[1:]) for class_label, centroid in centroids.items()}
        predicted_class = min(distances, key=distances.get)
        predictions.append(predicted_class)
    return predictions

def count_classes(test_data):
    classes = {}
    for row in test_data:
        classes[row[-1]] = classes.get(row[-1], 0) + 1
    return classes


# Évaluation du modèle grace à la précision
def evaluate(predictions, test_data):
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == test_data[i][-1]:
            correct += 1
    return (correct / len(predictions)) * 100

def result(best_train_data):
    best_train_data = sorted(best_train_data, key=lambda x: x[0])
    header = ['StudentID', 'Age', 'Gender', 'Ethnicity', 'ParentalEducation',	'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering',	'GradeClass']
    with open("best-data.csv", mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in best_train_data:
            writer.writerow(row)

if __name__ == '__main__':
    # Charger et convertir les données
    data = load_data('Student_performance_data _.csv')
    convert_data(data)

    best_accuracy = 0
    best_train_data = None

    classes = {}

    print("\n*******************************************************************************************************************************\n")
    print("CLASSIFICATION SUPERVISEE.\n")
   
    # Répéter 5 fois
    for _ in range(5):
        start_time = time.time()
        # Scinder les données
        train_data, test_data = split_data(data)

        # Calculer les centroïdes
        centroids = calculate_centroids(train_data)

        # Prédire la classe pour les données de test
        predictions = predict(test_data, centroids)

        # evaluer le temps de prédiction
        end_time = time.time()
        execution_time = end_time - start_time

        # compter les classes
        classes = count_classes(test_data)

        # Évaluer le modèle
        accuracy = evaluate(predictions, test_data)
        print(f'\n\nItération{_+1}')
        print(f'-> Précision du modèle : {accuracy:.2f}%')
        print(f'-> Temps d\'éxécution : {execution_time:.2f}s')
        for current_class, count in classes.items():
            print(f'-> Classe {current_class} : {count} occurrences')

        # Sauvegarder le meilleur jeu de données d'apprentissage
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_train_data = train_data

    # Afficher le meilleur jeu de données d'apprentissage
    result(best_train_data)
    print(f'\nMeilleure précision obtenue : {best_accuracy:.2f}%')
    print('\n\nMeilleur jeu de données d\'apprentissage :\n')
    for row in best_train_data:
        print(row)

    print("\n*******************************************************************************************************************************\n")




