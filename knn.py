import numpy as np


def knn(train_xs, train_cs, test_xs, test_cs):
    centroids = []
    for c in np.unique(cs):
        mask = cs == c
        c_xs = xs[mask]
        mean_c_xs = np.mean(c_xs, axis=0)
        # aggiunge i centroidi alla lista

    # per ogni test_xs, calcola la distanza euclidea
    # con tutti i centroidi e sceglie il più vicino
    # questa sara' la classe predetta
    # calcola l'accuracy, in percentuale, in base al
    # numero di predizioni corrette




def main():
    xs = np.random.rand(1000, 4)
    cs = np.random.randint(0, 3, (1000,))
    # suddivide i dati in train e test (80% train, 20% test)
    # (assiemen non li avevamo suddivisi)
    knn(train_xs, train_cs, test_xs, test_cs)
    
    
    



if __name__ == '__main__':
    main()
