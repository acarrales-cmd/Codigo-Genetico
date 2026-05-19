import random
import pandas as pd
import blosum
import copy
import time
import matplotlib.pyplot as plt

blosum62 = blosum.BLOSUM(62)
def get_sequences():
    seq1 = "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRDLYDDDDKDRWGKLVVLGAVTQGQKLVVLGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQV"
    seq2 = "MKTLLVAAAVVAGGQGQAEKLVKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQKELQKQLGQKAKEL"
    seq3 = "MAVTQGQKLVVLGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFAVVAGGQGQAEKLVKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKALCVFAIN"
    return [list(seq1), list(seq2), list(seq3)]

def crear_poblacion_inicial(n=10):
    individuo_base = get_sequences()
    return [[row[:] for row in individuo_base] for _ in range(n)]

def mutar_poblacion_v2(poblacion, num_gaps=1):
    poblacion_mutada = []
    for individuo in poblacion:
        nuevo_individuo = []
        for fila in individuo:
            fila_mutada = fila[:]
            posiciones = set()
            for _ in range(num_gaps):
                pos = random.randint(0, len(fila_mutada))
                while pos in posiciones:
                    pos = random.randint(0, len(fila_mutada))
                posiciones.add(pos)
                fila_mutada.insert(pos, '-')
            nuevo_individuo.append(fila_mutada)
        poblacion_mutada.append(nuevo_individuo)
    return poblacion_mutada

def igualar_longitud_secuencias(individuo, gap='-'):
    max_len = max(len(fila) for fila in individuo)
    return [fila + [gap]*(max_len - len(fila)) for fila in individuo]

def evaluar_individuo_blosum62(individuo):
    score = 0
    n_seqs = len(individuo)
    seq_len = len(individuo[0])
    for col in range(seq_len):
        for i in range(n_seqs):
            for j in range(i+1, n_seqs):
                a = individuo[i][col]
                b = individuo[j][col]
                if a == '-' or b == '-':
                    score -= 4
                else:
                    score += blosum62[a][b]
    return score

def eliminar_peores(poblacion, scores, porcentaje=0.5):
    idx_ordenados = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    n_seleccionados = int(len(poblacion) * porcentaje)
    ind_seleccionados = [poblacion[i] for i in idx_ordenados[:n_seleccionados]]
    scores_seleccionados = [scores[i] for i in idx_ordenados[:n_seleccionados]]
    return ind_seleccionados, scores_seleccionados

def mutar_individuo(individuo, n_gaps, p):
    nuevo_individuo = []
    for secuencia in individuo:
        sec = secuencia[:]
        if random.random() < p:
            posiciones = set()
            for _ in range(n_gaps):
                pos = random.randint(0, len(sec))
                while pos in posiciones:
                    pos = random.randint(0, len(sec))
                posiciones.add(pos)
                sec.insert(pos, '-')
        nuevo_individuo.append(sec)
    return nuevo_individuo

def cruzar_individuos_doble_punto(ind1, ind2):
    hijo1 = []
    hijo2 = []
    for seq1, seq2 in zip(ind1, ind2):
        aa_indices = [i for i, a in enumerate(seq1) if a != '-']
        if len(aa_indices) < 6:
            hijo1.append(seq1[:])
            hijo2.append(seq2[:])
            continue
        intentos = 0
        while True:
            p1, p2 = sorted(random.sample(aa_indices, 2))
            if p2 - p1 >= 5 or intentos > 10:
                break
            intentos += 1

        def cruza(seqA, seqB):
            aaA = [a for a in seqA if a != '-']
            aaB = [a for a in seqB if a != '-']
            nueva = aaA[:p1] + aaB[p1:p2] + aaA[p2:]
            resultado = []
            idx = 0
            for a in seqA:
                if a == '-':
                    resultado.append('-')
                else:
                    resultado.append(nueva[idx])
                    idx += 1
            return resultado

        hijo1.append(cruza(seq1, seq2))
        hijo2.append(cruza(seq2, seq1))

    hijo1 = mutar_individuo(hijo1, 1, 0.8)
    hijo2 = mutar_individuo(hijo2, 1, 0.8)
    return hijo1, hijo2

def validar_poblacion_sin_gaps(poblacion, originales):
    for individuo in poblacion:
        for seq, seq_orig in zip(individuo, originales):
            if [a for a in seq if a != '-'] != [a for a in seq_orig if a != '-']:
                return False
    return True

#Algoritmo original
def ejecutar_algoritmo_original(generaciones=150):
    poblacion = crear_poblacion_inicial(10)
    poblacion = mutar_poblacion_v2(poblacion, num_gaps=1)
    poblacion = [igualar_longitud_secuencias(ind) for ind in poblacion]
    
    historial_fitness = []
    veryBest = None
    fitnessVeryBest = float('-inf')

    for gen in range(generaciones):
        nueva_poblacion = []
        n = len(poblacion)
        indices = list(range(n))
        random.shuffle(indices)
        parejas = [(indices[i], indices[i+1]) for i in range(0, n-1, 2)]
        if n % 2 == 1:
            parejas.append((indices[-1], indices[0]))
        for idx1, idx2 in parejas:
            padre1 = poblacion[idx1]
            padre2 = poblacion[idx2]
            hijo1, hijo2 = cruzar_individuos_doble_punto(padre1, padre2)
            nueva_poblacion.extend([copy.deepcopy(padre1), copy.deepcopy(padre2), hijo1, hijo2])
        
        poblacion = [igualar_longitud_secuencias(ind) for ind in nueva_poblacion[:2*n]]
        scores = [evaluar_individuo_blosum62(ind) for ind in poblacion]
        poblacion, scores = eliminar_peores(poblacion, scores, porcentaje=0.5)
        
        idx_mejor = scores.index(max(scores))
        if scores[idx_mejor] > fitnessVeryBest:
            fitnessVeryBest = scores[idx_mejor]
            veryBest = copy.deepcopy(poblacion[idx_mejor])
        historial_fitness.append(fitnessVeryBest)

    return historial_fitness, veryBest


# Algoritmo modificado
def ejecutar_algoritmo_mejorado_seguro(generaciones=150):
    poblacion = crear_poblacion_inicial(30)
    poblacion = mutar_poblacion_v2(poblacion, num_gaps=1)
    poblacion = [igualar_longitud_secuencias(ind) for ind in poblacion]
    
    historial_fitness = []
    veryBest = None
    fitnessVeryBest = float('-inf')

    for gen in range(generaciones):
        scores = [evaluar_individuo_blosum62(ind) for ind in poblacion]
        
        idx_mejor = scores.index(max(scores))
        if scores[idx_mejor] > fitnessVeryBest:
            fitnessVeryBest = scores[idx_mejor]
            veryBest = copy.deepcopy(poblacion[idx_mejor])
        historial_fitness.append(fitnessVeryBest)

        nueva_poblacion_hijos = []
        while len(nueva_poblacion_hijos) < 40:
            i1, i2 = random.sample(range(len(poblacion)), 2)
            padre1 = poblacion[i1] if scores[i1] > scores[i2] else poblacion[i2]
            
            i3, i4 = random.sample(range(len(poblacion)), 2)
            padre2 = poblacion[i3] if scores[i3] > scores[i4] else poblacion[i4]
            
            hijo1, hijo2 = cruzar_individuos_doble_punto(padre1, padre2)
            nueva_poblacion_hijos.extend([copy.deepcopy(padre1), copy.deepcopy(padre2), hijo1, hijo2])
            
        nueva_poblacion_hijos = [igualar_longitud_secuencias(ind) for ind in nueva_poblacion_hijos[:60]]
        scores_hijos = [evaluar_individuo_blosum62(ind) for ind in nueva_poblacion_hijos]
        
        poblacion, scores_filtrados = eliminar_peores(nueva_poblacion_hijos, scores_hijos, porcentaje=0.5)
        
        idx_peor = scores_filtrados.index(min(scores_filtrados))
        poblacion[idx_peor] = copy.deepcopy(veryBest)

    return historial_fitness, veryBest

if __name__ == "__main__":
    print("Ejecutando Algoritmo Original de la Tarea...")
    hist_orig, best_orig = ejecutar_algoritmo_original(150)
    
    print("Ejecutando Algoritmo Optimizado (Torneo + Elitismo)...")
    hist_mej, best_mej = ejecutar_algoritmo_mejorado_seguro(150)

    print("\n--- EVALUACIÓN DE ENTREGA ---")
    print(f"Mejor Fitness Encontrado por Original: {hist_orig[-1]}")
    print(f"Mejor Fitness Encontrado por Optimizado: {hist_mej[-1]}")
    print("Validación Integridad Estricta (Original):", validar_poblacion_sin_gaps([best_orig], get_sequences()))
    print("Validación Integridad Estricta (Optimizado):", validar_poblacion_sin_gaps([best_mej], get_sequences()))

    # Gráfica 1: Original
    plt.figure(figsize=(10, 5))
    plt.plot(hist_orig, label="Algoritmo Original", color="red", linestyle="--")
    plt.title("Rendimiento del Algoritmo Original")
    plt.xlabel("Generación")
    plt.ylabel("Fitness (Score BLOSUM62)")
    plt.legend(); plt.grid(True)
    plt.savefig("grafica_original.png"); plt.close()

    # Gráfica 2: Mejorado
    plt.figure(figsize=(10, 5))
    plt.plot(hist_mej, label="Algoritmo Optimizado", color="blue", linewidth=2)
    plt.title("Rendimiento del Algoritmo Optimizado (Torneo + Elitismo)")
    plt.xlabel("Generación")
    plt.ylabel("Fitness (Score BLOSUM62)")
    plt.legend(); plt.grid(True)
    plt.savefig("grafica_mejorado.png"); plt.close()

    # Gráfica 3: Comparativa
    plt.figure(figsize=(10, 5))
    plt.plot(hist_orig, label="Algoritmo Original", color="red", linestyle="--")
    plt.plot(hist_mej, label="Algoritmo Optimizado", color="blue", linewidth=2)
    plt.title("Comparación de Alineamiento: Original vs Optimizado")
    plt.xlabel("Generación")
    plt.ylabel("Fitness (Score BLOSUM62)")
    plt.legend(); plt.grid(True)
    plt.savefig("comparacion_fitness.png"); plt.close()
    print("\n¡Las 3 gráficas independientes se han generado con éxito!")
