import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.optimize import shgo
from scipy.optimize import dual_annealing
from scipy.optimize import dual_annealing
from scipy.optimize import direct

a = .3
P = 30e3
Sadm = 248.2e6
ro = 7861
hmax = 36*a/350
bounds = ([0, 31], [0, 31], [0, 31], [0, 31], [0, 31], [0, 31], [32, 50], [32, 50], [32, 50])

# [p1, p2, p3, p4, p5, p6, v1, v2, v3]
# Matriz de Seções: [número da seção, área, módulo de elasticidade, momento de inércia, 

no = [   0,       1,       2,       3,       4,             5,       6,       7,          8,       9,       10,       11]
x  = [60*a,       0,       0,       0,       0,          30*a,    30*a,    30*a,       30*a,    60*a,       60*a,   60*a]
y  = [36*a,       0,       12*a,    24*a,       36*a,       0,    12*a,    24*a,       36*a,       0,       12*a,   24*a]

def portico (projeto):

    p1 = projeto[0]
    p2 = projeto[1]
    p3 = projeto[2]
    p4 =  projeto[3]
    p5 = projeto[4]
    p6 = projeto[5]
    v1 = projeto[6]
    v2 = projeto[7]
    v3 = projeto[8]
    
    projeto = [p1, p2, p3, p4, p5, p6, v1, v2, v3]
    
    secoes = np.array([[0, 0.0573, 200.0e9, 0.00878, 0.4735], 
                    [1, 0.0256, 200.0e9, 0.003250, 0.452], 
                    [2, 0.0382, 200.0e9, 0.004830, 0.428], 
                    [3, 0.0224, 200.0e9, 0.002460, 0.418], 
                    [4, 0.0329, 200.0e9, 0.003430, 0.386], 
                    [5, 0.01880, 200.0e9, 0.001660, 0.377], 
                    [6, 0.027800, 200.0e9, 0.002360, 0.348], 
                    [7, 0.01600, 200.0e9, 0.001190, 0.339], 
                    [8, 0.019700, 200.0e9, 0.001290, 0.306], 
                    [9, 0.01300, 200.0e9, 0.000762, 0.301],    # W610x101
                    [10, 0.019200, 200.0e9, 0.001010, 0.272], 
                    [11, 0.011800, 200.0e9, 0.000554, 0.2665], # W530x92
                    [12, 0.00839, 200.0e9, 0.000351, 0.263], 
                    [13, 0.020100, 200.0e9, 0.000795, 0.2375], 
                    [14, 0.01440, 200.0e9, 0.000554, 0.231], 
                    [15, 0.009480, 200.0e9, 0.000333, 0.2285], 
                    [16, 0.006650, 200.0e9, 0.000212, 0.225], 
                    [17, 0.014600, 200.0e9, 0.000462, 0.2095], 
                    [18, 0.010800, 200.0e9, 0.000316, 0.2085], 
                    [19, 0.007610, 200.0e9, 0.000216, 0.203], 
                    [20, 0.005890, 200.0e9, 0.000156, 0.202],  
                    [21, 0.004950, 200.0e9, 0.000125, 0.1995], 
                    [22, 0.07030, 200.0e9, 0.002260, 0.2275], 
                    [23, 0.0275, 200.0e9, 0.000712, 0.188], 
                    [24, 0.0155, 200.0e9, 0.000367, 0.1815], 
                    [25, 0.0129, 200.0e9, 0.000301, 0.178], 
                    [26, 0.0101, 200.0e9, 0.000225, 0.1765], 
                    [27, 0.008130, 200.0e9, 0.000178, 0.174], 
                    [28, 0.007230, 200.0e9, 0.000160, 0.179], 
                    [29, 0.005710, 200.0e9, 0.000121, 0.1755], 
                    [30, 0.00496, 200.0e9, 0.000102, 0.1765], 
                    [31, 0.004190, 200.0e9, 0.0000828, 0.174], 
                    [32, 0.0182, 200e9, 0.000347, 0.323/2], ## Perfis de PILAR Daqui para baixo  W310x143
                    [33, 0.0136, 200e9, 0.000248, 0.312/2],   # W310x107
                    [34, 0.00942, 200e9, 0.000163, 0.31/2],   # W310x74
                    [35, 0.00755, 200e9, 0.000128, 0.302/2], 
                    [36, 0.00665, 200e9, 0.000119, 0.318/2],  # W310x52
                    [37, 0.00567, 200e9, 0.0000991, 0.312/2], 
                    [38, 0.00494, 200e9, 0.0000849, 0.31/2], 
                    [39, 0.00418, 200e9, 0.0000649, 0.312/2], # W310x32.7
                    [40, 0.00304, 200e9, 0.0000429, 0.305/2], 
                    [41, 0.0212, 200e9, 0.000298 ,0.29/2], 
                    [42, 0.0129, 200e9, 0.000164,0.264/2], 
                    [43, 0.0102, 200e9, 0.000126,0.257/2], 
                    [44, 0.00858, 200e9, 0.000103,0.257/2], 
                    [45, 0.00742, 200e9, 0.000087,0.252/2], 
                    [46, 0.00626, 200e9, 0.0000712,0.247/2], 
                    [47, 0.0057, 200e9, 0.0000708,0.267/2], 
                    [48, 0.00419, 200e9, 0.0000491,0.259/2], 
                    [49, 0.00363, 200e9, 0.0000401,0.259/2], 
                    [50, 0.00285, 200e9, 0.0000287,0.254/2]])

    n_sec = 9  # Número de seções distintas presentes na estrutura
    n_el = 15
    n_nos = 12

    # Matriz de conectividade: [elemento, Número da seção, primeiro nó, segundo nó]
    conec = np.array([[0,   v3,   8,   0],
                    [1,   p1,   1,   2],
                    [2,   p4,   5,   6],
                    [3,   p1,   9,   10],
                    [4,   p2,   2,   3],
                    [5,   p5,   6,   7],
                    [6,   p2,   10,   11],
                    [7,   p3,   3,   4],
                    [8,   p6,   7,   8],
                    [9,   p3,   11,   0],
                    [10,   v1,   2,   6],
                    [11,   v1,   6,   10],
                    [12,   v2,   3,   7],
                    [13,   v2,   7,   11],
                    [14,   v3,   4,   8],])

    # %%
    # Carregamentos nodais (Fzão da estrutura)
    n_forcas = 3  # Número de nós na qual atuam forças
    # Matriz de forças [nó (primeiro nó é o nó zero e não 1), força em x, força em y, momento]
    forca_nodal = 30 * (10**3)
    forcas = np.matrix(
        [[2, forca_nodal, 0, 0], [3, forca_nodal, 0, 0], [4, forca_nodal, 0, 0]]
    )

    # %%
    # Carregamentos equivalentes (Feq da estrutura)
    n_eq = 6  # número de elementos que contem carregamentos equivalentes
    # Matriz de carregamento equivalente = [elemento, tipo de carregamento, intensidade, posição (para o caso de carregamento concentrado entre nós)]
    carreg_uniforme = 18 * (10**3)
    w_eq = np.array(
        [
            [10, 1, -carreg_uniforme, 0],
            [11, 1, -carreg_uniforme, 0],
            [12, 1, -carreg_uniforme, 0],
            [13, 1, -carreg_uniforme, 0],
            [14, 1, -carreg_uniforme, 0],
            [0, 1, -carreg_uniforme, 0],
        ]
    )
    # LEMBRETE: os sinais das forças devem seguir o sistema LOCAL do elemento!

    # %%
    # Apoios
    n_rest = 3  # número de nós restringidos
    # Matriz de condições de contorno
    # [número do nó, restringido_x, restringido_y, restringido_theta] (1 para restringido, e 0 para livre)
    GDL_rest = np.array([[1, 1, 1, 0], [5, 1, 1, 0], [9, 1, 1, 0]])


    # %%
    # CALCULO DA ESTRUTURA
    GDL = 3 * n_nos  # graus de liberdade da estrutura
    K = np.zeros((GDL, GDL))  # matriz rigidez global

    # Cálculo da matriz de cada elemento
    for el in range(n_el):
        # print(el)
        # calculo do comprimento do elemento el
        no1 = int(conec[el, 2])
        no2 = int(conec[el, 3])
        # L=abs(x(no2)-x(no1))
        L = (np.sqrt((x[int(no2)] - x[int(no1)]) ** 2 + (y[int(no2)] - y[int(no1)]) ** 2))
        # Propriedades
        j = int(conec[int(el)][1])
        A = secoes[j][1]
        E = secoes[j][2]
        Iz = secoes[j][3]
        # Cossenos diretores a partir das coordenadas dos ns do elemento
        c = (x[int(no2)] - x[int(no1)]) / L  # cosseno
        s = (y[int(no2)] - y[int(no1)]) / L  #  seno
        # Matriz de transformação do elemento "el"
        T = np.array(
            [
                [c, s, 0, 0, 0, 0],
                [-s, c, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, c, s, 0],
                [0, 0, 0, -s, c, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        # Construo da matriz de rigidez em coordenadas locais
        k1 = E * A / L
        k2 = 12 * E * Iz / L**3
        k3 = 6 * E * Iz / L**2
        k4 = 4 * E * Iz / L
        k5 = k4 / 2
        k = np.array(
            [
                [k1, 0, 0, -k1, 0, 0],
                [0, k2, k3, 0, -k2, k3],
                [0, k3, k4, 0, -k3, k5],
                [-k1, 0, 0, k1, 0, 0],
                [0, -k2, -k3, 0, k2, -k3],
                [0, k3, k5, 0, -k3, k4],
            ]
        )
        # Matriz de rigidez em coordenadas globais
        kg = np.dot(np.transpose(T), np.dot(k, T))

        # Determinando matriz de incidência cinemática:
        b = np.zeros((6, GDL))
        i = int(no1)
        j = int(no2)
        b[0, 3 * i] = 1
        b[1, 3 * i + 1] = 1
        b[2, 3 * i + 2] = 1
        b[3, 3 * j] = 1
        b[4, 3 * j + 1] = 1
        b[5, 3 * j + 2] = 1
        # Expandindo e convertendo a matriz do elemento para coordenadas globais:
        Ki = np.dot(np.transpose(b), np.dot(kg, b))
        # Somando contribuição do elemento para a matriz de rigidez global:
        K = K + Ki

    # %%
    # Vetor de forcas Global
    F = np.zeros((GDL, 1))
    for i in range(n_forcas):
        F[int(3 * forcas[i, 0])] = forcas[i, 1]
        F[int(3 * forcas[i, 0]) + 1] = forcas[i, 2]
        F[int(3 * forcas[i, 0]) + 2] = forcas[i, 3]

    # %%
    # Construção do vetor de foras equivalentes
    Feq = np.zeros((GDL, 1))
    for i in range(n_eq):
        tipo = int(w_eq[i, 1])  # tipo de força equivalente
        el = int(w_eq[i, 0])  # elemento onde está aplicada
        if tipo == 1:  # Carregamento distribuído
            f = np.zeros((6, 1))
            no1 = int(conec[el, 2])
            no2 = int(conec[el, 3])
            L = np.sqrt((x[no2] - x[no1]) ** 2 + (y[no2] - y[no1]) ** 2)
            w = w_eq[i, 2]
            f[0] = 0
            f[1] = +w * L / 2
            f[2] = +w * L**2 / 12
            f[3] = 0
            f[4] = +w * L / 2
            f[5] = -w * L**2 / 12
            # Cossenos diretores a partir das coordenadas dos ns do elemento
            c = (x[no2] - x[no1]) / L  # cosseno
            s = (y[no2] - y[no1]) / L  #  seno
            # Matriz de transformação do elemento "el"
            T = np.array(
                [
                    [c, s, 0, 0, 0, 0],
                    [-s, c, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, c, s, 0],
                    [0, 0, 0, -s, c, 0],
                    [0, 0, 0, 0, 0, 1],
                ]
            )
            # feqTT=np.dot(np.transpose(T),f)
            feq = np.matmul(np.transpose(T), f)
            Feq[3 * no1] = Feq[3 * no1] + feq[0]
            Feq[3 * no1 + 1] = Feq[3 * no1 + 1] + feq[1]
            Feq[3 * no1 + 2] = Feq[3 * no1 + 2] + feq[2]
            Feq[3 * no2] = Feq[3 * no2] + feq[3]
            Feq[3 * no2 + 1] = Feq[3 * no2 + 1] + feq[4]
            Feq[3 * no2 + 2] = Feq[3 * no2 + 2] + feq[5]
        elif tipo == 2:  ## carga aplicada a uma distancia a do nó i
            f = np.zeros((6, 1))
            no1 = int(conec[el, 2])
            no2 = int(conec[el, 3])
            L = np.sqrt((x[no2] - x[no1]) ** 2 + (y[no2] - y[no1]) ** 2)
            a = w_eq[i, 3]
            b = L - a
            p = w_eq[i, 2]
            f[0] = 0
            f[1] = +p * b**2 * (3 * a + b) / L**3
            f[2] = +p * a * b**2 / L**2
            f[3] = 0
            f[4] = +p * a**2 * (a + 3 * b) / L**3
            f[5] = -p * a**2 * b / L**2
            # Cossenos diretores a partir das coordenadas dos nós do elemento
            c = (x[no2] - x[no1]) / L  # cosseno
            s = (y[no2] - y[no1]) / L  #  seno
            # Matriz de transformação do elemento "el"
            T = np.array(
                [
                    [c, s, 0, 0, 0, 0],
                    [-s, c, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, c, s, 0],
                    [0, 0, 0, -s, c, 0],
                    [0, 0, 0, 0, 0, 1],
                ]
            )
            # feqTT=np.dot(np.transpose(T),f)
            feq = np.matmul(np.transpose(T), f)
            Feq[3 * no1] = Feq[3 * no1] + feq[0]
            Feq[3 * no1 + 1] = Feq[3 * no1 + 1] + feq[1]
            Feq[3 * no1 + 2] = Feq[3 * no1 + 2] + feq[2]
            Feq[3 * no2] = Feq[3 * no2] + feq[3]
            Feq[3 * no2 + 1] = Feq[3 * no2 + 1] + feq[4]
            Feq[3 * no2 + 2] = Feq[3 * no2 + 2] + feq[5]

    # %%
    # guardamos os originais de K e F
    Kg = np.copy(K)
    # Kg[:] = K[:]

    Fg = F + Feq
    # Aplicar Restrições (condições de contorno)
    for k in range(n_rest):
        # Verifica se há restrição na direção x
        if GDL_rest[k, 1] == 1:
            j = 3 * GDL_rest[k, 0]
            # Modificar Matriz de Rigidez
            for i in range(GDL):
                Kg[j, i] = 0  # zera linha
                Kg[i, j] = 0  # zera coluna
            Kg[j, j] = 1  # valor unitário na diagonal principal
            Fg[j] = 0
        # Verifica se há restrição na direção y
        if GDL_rest[k, 2] == 1:
            j = 3 * GDL_rest[k, 0] + 1
            # Modificar Matriz de Rigidez
            for i in range(GDL):
                Kg[j, i] = 0  # zera linha
                Kg[i, j] = 0  # zera coluna
            Kg[j, j] = 1  # valor unitário na diagonal principal
            Fg[j] = 0
        # Verifica se há restrição na rotação
        if GDL_rest[k, 3] == 1:
            j = 3 * GDL_rest[k, 0] + 2
            # Modificar Matriz de Rigidez
            for i in range(GDL):
                Kg[j, i] = 0  # zera linha
                Kg[i, j] = 0  # zera coluna
            Kg[j, j] = 1  # valor unitário na diagonal principal
            Fg[j] = 0

    # %%
    # Calculo dos deslocamentos
    desloc = np.linalg.solve(Kg, Fg)

    # %%
    # Reações
    reacoes = np.matmul(K, desloc) - Feq
    # reacoes=K*desloc-Feq

    # %%
    # Esforços nos elementos
    f_el = np.zeros((n_el, 6))
    N = np.zeros((n_el, 1))
    Mmax = np.zeros((n_el, 1))
    Smax = np.zeros((n_el, 1))
    Falha = np.zeros((n_el, 1))
    Sadm = 248.2e6  # colocar valor do pdf do projeto
    peso = 0
    ro = 7861
    for el in range(n_el):
        # calculo do comprimento do elemento el
        no1 = int(conec[el, 2])
        no2 = int(conec[el, 3])
        # L=abs(x(no2)-x(no1))
        L = np.sqrt((x[no2] - x[no1]) ** 2 + (y[no2] - y[no1]) ** 2)
        # Propriedades
        j = int(conec[el, 1])
        A = secoes[j, 1]
        E = secoes[j, 2]
        Iz = secoes[j, 3]
        cc = secoes[j, 4]
        # calculo peso
        peso = peso + A * L * ro
        # Cossenos diretores a partir das coordenadas dos ns do elemento
        c = (x[no2] - x[no1]) / L  # cosseno
        s = (y[no2] - y[no1]) / L  #  seno
        # Matriz de transformação do elemento "el"
        T = np.array(
            [
                [c, s, 0, 0, 0, 0],
                [-s, c, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, c, s, 0],
                [0, 0, 0, -s, c, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        # Construção da matriz de rigidez em coordenadas locais
        k1 = E * A / L
        k2 = 12 * E * Iz / L**3
        k3 = 6 * E * Iz / L**2
        k4 = 4 * E * Iz / L
        k5 = k4 / 2
        ke = np.array(
            [
                [k1, 0, 0, -k1, 0, 0],
                [0, k2, k3, 0, -k2, k3],
                [0, k3, k4, 0, -k3, k5],
                [-k1, 0, 0, k1, 0, 0],
                [0, -k2, -k3, 0, k2, -k3],
                [0, k3, k5, 0, -k3, k4],
            ]
        )
        # pega os valores dos deslocamentos dos nós do elemento "el"
        u1 = desloc[no1 * 3]
        u2 = desloc[no2 * 3]
        v1 = desloc[no1 * 3 + 1]
        v2 = desloc[no2 * 3 + 1]
        th1 = desloc[no1 * 3 + 2]
        th2 = desloc[no2 * 3 + 2]
        d_g = np.array([u1, v1, th1, u2, v2, th2])
        d_el = np.matmul(T, d_g)
        # d_el=T*d_g

        ## forças equivalentes: recalcula vetor de feq. no sistema local
        aux = []
        cont = [0]
        for temp in w_eq[:, 0]:
            if int(temp) == el:
                aux = cont[:]
            cont[0] = cont[0] + 1
        if len(aux) == 0:
            feqq = 0
        else:
            aux = int(aux[0])
            tipo = w_eq[aux, 1]  # tipo de força equivalente
            if tipo == 1:
                w = w_eq[aux, 2]
                feqq = np.zeros((6, 1))
                feqq[0] = 0
                feqq[1] = +w * L / 2
                feqq[2] = +w * L**2 / 12
                feqq[3] = 0
                feqq[4] = +w * L / 2
                feqq[5] = -w * L**2 / 12
            elif tipo == 2:
                a = w_eq[aux, 3]
                b = L - a
                p = w_eq[aux, 2]
                feqq = np.zeros((6, 1))
                feqq[0] = 0
                feqq[1] = +p * b**2 * (3 * a + b) / L**3
                feqq[2] = +p * a * b**2 / L**2
                feqq[3] = 0
                feqq[4] = +p * a**2 * (a + 3 * b) / L**3
                feqq[5] = -p * a**2 * b / L**2

        ## esforços locais atuantes no elemento "el": cada linha da matriz f_el
        # contem os esforços de um elemento = [fx_1' fy_1' mz_1' fx_2' fy_2' mz_2']
        f_el[el, :] = np.transpose(np.matmul(ke, d_el) - feqq)
        # Esforços para cálculo de tensão
        N = abs(f_el[el, 0])
        Mzi = abs(f_el[el, 2])
        Mzj = abs(f_el[el, 5])
        if el > 0 and el < 10:
            aux = np.array([Mzi, Mzj])
            Mmax[el] = aux.max()
        else:
            Mvao = -f_el[el, 2] + f_el[el, 1] / (-2 * w)
            aux = np.array([Mzi, Mzj, Mvao])
            Mmax[el] = aux.max()

        # Cálculo da tensão
        Smax[el] = N / A + Mmax[el] / Iz * cc

        # Critério de Falha
        if Smax[el] > Sadm:
            Falha[el] = 1

    #check_fail = np.where(Falha == [1])
    #is_failed = len(check_fail[0]) > 0
    
    score = float (sum(Falha) * 10e8 + peso)

    return (score, peso)

result = differential_evolution(portico, bounds)

result1 = shgo(portico, bounds)

result2 = dual_annealing(portico, bounds, maxiter=100)

print("Differential Evolution")
print('Resultado: ', result)

print("\n")

print("SHGO")
print('Resultado: ', result1)

print("\n")
print("Dual Annealing")
print('Resultado: ', result2)

