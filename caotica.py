# encoding: utf-8

import numpy as np

def caotica(N, p, A0) :
    # Série caótica a partir do mapeamento quadrático
    #   A(n+1) = p . A(n) . ( 1 - A(n) )
    # Parâmetros:
    #   N = 2**12 (0..N-1), p = 4.0, A0 = 0.001
    # Uso:
    #  import caotica as ca
    #  x = ca.caot(2**12, 4.0, 0.001)
    A = np.zeros((N), dtype=float)  # cria o array
    A[0] = A0                       # condicao inicial do mapeamento
    for i in range(1, N-1) :        # demais numeros
        A[i] = p * A[i-1] * ( 1 - A[i-1] )
    return A