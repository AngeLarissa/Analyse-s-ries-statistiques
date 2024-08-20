import csv

# Fonction pour lire et filtrer les données du fichier CSV
def read_csv_file(nom_fichier):
    with open(nom_fichier, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader if int(row['Season'][:4]) >= 2012 and int(row['Season'][:4]) <= 2022]
    return data

# Étape 1: Sélectionner tous les clubs ayant joué en Premier League de 2012 à 2022
def select_clubs(data):
    clubs = set()
    for row in data:
        clubs.add(row['Team'])
    return list(clubs)

# Étape 2: Compter le nombre d'occurrences de chaque club en Premier League de 2012 à 2022
def count_occurrences(data):
    occurrences = {}
    for row in data:
        team = row['Team']
        if team in occurrences:
            occurrences[team] += 1
        else:
            occurrences[team] = 1
    return occurrences

# Étape 3: Classer les clubs par ordre décroissant d'apparition en Premier League de 2012 à 2022
def order_clubs(occurrences):
    sorted_clubs = sorted(occurrences.items(), key=lambda x: x[1], reverse=True)
    return [club[0] for club in sorted_clubs]

# Étape 4: Conserver les 25 premiers clubs
def keep_clubs(clubs, n=25):
    return clubs[:n]

# Étape 5: Produire un fichier CSV avec les résultats
def result(clubs, data):
    years = list(range(2012, 2023))
    header = ['Club'] + [str(year) for year in years]
    
    with open('BOKOU.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        
        for club in clubs:
            row = [club]
            for year in years:
                points = 0
                for entry in data:
                    if entry['Team'] == club and entry['Season'][:4] == str(year):
                        points = int(entry['Pts'])
                        break
                row.append(points)
            writer.writerow(row)

def read_csv_file2(nom_fichier):
    with open(nom_fichier, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        rows = [row for row in reader]
    data = []
    observed_values = []
    for row in rows:
        values = list(row.values())
        a = values[1:-2]
        a.insert(0, 1)
        a = [int(value) for value in a]
        data.append(a)
        b = values[-2:-1]
        b = [int(value) for value in b]
        observed_values.append(b)
    return data, observed_values
    

def matrix_multiply(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    # vérifier si A et B peuvent etre multipliés.
    if cols_A != rows_B:
        raise ValueError("Number of columns in A must be equal to number of rows in B")

    # initialiser la matrice résultante à 0
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    # matrix multiplication
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]

    return result

def transpose_matrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    # initialiser la transposée à 0
    transposed = [[0 for _ in range(rows)] for _ in range(cols)]

    # transposition
    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]

    return transposed



def gauss_jordan_inverse(matrix):
    n = len(matrix)
    # matrice idnetité de la meme taille
    identity_matrix = [[float(i == j) for i in range(n)] for j in range(n)]

    # construire la matrice augmentée (matrice initiale + matrice identité)
    for i in range(n):
        matrix[i] += identity_matrix[i]

    # méthode de Gauss-Jordan 
    for i in range(n):
        # transformer tous les éléments de la diagonale en 1
        factor = matrix[i][i]
        for j in range(2 * n):
            matrix[i][j] /= factor
        
        # transformer les éléments des autres colonnes en 0
        for k in range(n):
            if k != i:
                factor = matrix[k][i]
                for j in range(2 * n):
                    matrix[k][j] -= factor * matrix[i][j]

    # extraire la partie droite de la matrice augmentée: c'est l'inverse obtenue
    inverse_matrix = [row[n:] for row in matrix]
    return inverse_matrix


def find_matrix_parameters(data, observed_values):
    transposed_data = transpose_matrix(data)
    matrix1 = gauss_jordan_inverse(matrix_multiply(transposed_data, data))
    matrix2 = matrix_multiply(transposed_data, observed_values)
    return (matrix_multiply(matrix1, matrix2))

def find_estimated_values(data, parameters):
    estimated_values = []
    for row in data:
        value = 0
        for i in range(1, len(row)):
            value += parameters[i][0] * row[i]
        value += parameters[0][0]
        estimated_values.append(value)
    return estimated_values

def calculate_mean(observed_values):
    summ = 0
    for row in observed_values:
        summ += row[0]
    return summ / (len(observed_values))


def find_scr(observed_values, estimated_values):
    scr = 0
    for i in range(len(observed_values)):
        scr += ((observed_values[i][0] - estimated_values[i]) ** 2)
    return scr

def find_sce(estimated_values, mean):
    sce = 0
    for i in range(len(estimated_values)):
        sce += ((mean - estimated_values[i]) ** 2)
    return sce

def find_sct(scr, sce):
    return scr + sce

def find_coefficient(sce, sct):
    return (sce / sct) * 100

def set_conclusion(coefficient):
    print("\n*******************************************************************************************************************************\n")
    print(("Evaluation du modèle  de regression linéaire utilisée pour étudier les points des équipes en Premier League.\n").upper())
    print(f"Ce modèle de regression explique {coefficient:.2f}% de la variation totale des valeurs des points obtenus par les équipes en 2021.")
    print("\n*******************************************************************************************************************************\n")

        
if __name__ == "__main__":
    file = "EPL_Standings_2000-2022.csv"
    data = read_csv_file(file)
    clubs = select_clubs(data)
    occurrences = count_occurrences(data)
    ordered_clubs = order_clubs(occurrences)
    result_clubs = keep_clubs(ordered_clubs, n=25)
    result(result_clubs, data)
    data, observed_values = read_csv_file2('BOKOU.csv')
    parameters = find_matrix_parameters(data, observed_values)
    estimated_values = find_estimated_values(data, parameters)
    mean = calculate_mean(observed_values)
    scr = find_scr(observed_values, estimated_values)
    sce = find_sce(estimated_values, mean)
    sct = find_sct(scr, sce)
    coefficient = find_coefficient(sce, sct)
    set_conclusion(coefficient)
 
   
