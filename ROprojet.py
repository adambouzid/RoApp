

import tkinter as tk
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
##from matplotlib.colors import Normalize
#import matplotlib.cm as cm
import matplotlib
from tkinter import simpledialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

matplotlib.use('Qt5Agg')  # Utiliser le backend TkAgg pour Matplotlib

def open_algo_menu():
    algo_window = tk.Toplevel(gui)
    algo_window.title("Menu")
    algo_window.geometry("600x800")  # Augmentation de la taille pour afficher tous les boutons
    algo_window.configure(bg="#121212")  # Fond sombre intense et agressif

    # Label principal avec un style futuriste et percutant
    menu_label = tk.Label(
        algo_window, text="Veuillez sélectionner un algorithme", 
        font=("Arial", 22, "bold"), fg="#ffffff", bg="#121212"
    )
    menu_label.pack(pady=20)  # Réduction de l'espace pour plus d'ajustement

    # Liste des algorithmes avec des boutons stylisés et percutants
    algo_list = [
        ("Welsh Powell", execute_welshpowell),
        ("Dijkstra", execute_dijkstra),
        ("Kruskal", execute_kruskal),
        ("Bellman Ford", execute_bellmanford),
        ("Potentiel-Metra", execute_potentiel_metra),
        ("Moindre Coût", execute_moindre_cout),
        ("Nord-Ouest", execute_nord_ouest),
        ("Stepping Stone", execute_stepping_stone),
        ("Ford Fulkerson", execute_ford_fulkerson),
    ]

    # Création des boutons pour chaque algorithme avec un design ovale
    for name, command in algo_list:
        tk.Button(
            algo_window, text=name, command=command,
            font=("Arial", 14, "bold"), bg="#ff4444", fg="white",
            activebackground="#ff6a6a", activeforeground="black",  # Rouge agressif avec un survol encore plus vif
            width=30, height=2, bd=4, relief="solid",  # Bordure nette et contrastée
            cursor="hand2",
            highlightthickness=0,  # Suppression de l'outline
            borderwidth=0,  # Suppression de la bordure interne
        ).pack(pady=10)  # Ajustement de l'espace entre les boutons

    # Bouton retour avec un design ovale
    exit_button = tk.Button(
        algo_window, text="Retour", command=algo_window.destroy,
        font=("Arial", 16, "bold"), bg="#d32f2f", fg="#ffffff",
        activebackground="#b71c1c", activeforeground="white",  # Rouge vif pour le bouton retour
        width=30, height=2, bd=4, relief="solid",  # Bordure nette
        cursor="hand2",
        highlightthickness=0,  # Suppression de l'outline
        borderwidth=0,  # Suppression de la bordure interne
    )
    exit_button.pack(side="bottom", pady=20)  # Réduction de l'espace pour éviter de déborder


# Algorithme de Welsh-Powell
def welsh_powell_coloring(graph):
    sorted_nodes = sorted(graph.nodes, key=lambda x: graph.degree[x], reverse=True)
    colors = {}
    current_color = 0

    for node in sorted_nodes:
        if node not in colors:
            current_color += 1
            colors[node] = current_color
            for other_node in sorted_nodes:
                if other_node not in colors and all(colors.get(neighbor) != current_color for neighbor in graph.neighbors(other_node)):
                    colors[other_node] = current_color
    return colors, current_color


def execute_welshpowell():
    try:
        num_nodes = int(simpledialog.askstring("Entrée", "Entrez le nombre de sommets du graphe :"))
        num_edges = int(simpledialog.askstring("Entrée", "Entrez le nombre d'arêtes du graphe :"))
        if num_edges < num_nodes - 1:
            raise ValueError("Le nombre d'arêtes doit être au moins égal à (n - 1) pour un graphe connexe.")
    except (ValueError, TypeError):
        messagebox.showerror("Erreur", "Entrées invalides. Veuillez entrer des entiers valides.")
        return

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))

    # Générer un arbre couvrant minimal pour garantir la connexité
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    for i in range(len(nodes) - 1):
        graph.add_edge(nodes[i], nodes[i + 1])

    # Ajouter des arêtes supplémentaires aléatoires
    edges = set(graph.edges)
    while len(edges) < num_edges:
        u, v = random.sample(range(num_nodes), 2)
        if u != v and (u, v) not in edges and (v, u) not in edges:
            edges.add((u, v))
            graph.add_edge(u, v)

    colors, chromatic_number = welsh_powell_coloring(graph)

    result = f"Nombre chromatique (Chromatic Number) : {chromatic_number}\n"
    result += "\n".join([f"Sommets {node} -> Couleur {color}" for node, color in colors.items()])
    messagebox.showinfo("Résultats Welsh-Powell", result)

    pos = nx.spring_layout(graph)
    color_map = [colors[node] for node in graph.nodes]
    plt.figure(figsize=(8, 6))
    nx.draw(
        graph, pos, with_labels=True, node_color=color_map,
        node_size=500, cmap=plt.cm.rainbow, font_color='white'
    )
    plt.title(f"Coloriage du graphe (Nombre chromatique : {chromatic_number})")
    plt.show()


# Algorithme de Dijkstra
def execute_dijkstra():
    try:
        num_nodes = int(simpledialog.askstring("Entrée", "Entrez le nombre de sommets du graphe :"))
        num_edges = int(simpledialog.askstring("Entrée", "Entrez le nombre d'arêtes du graphe :"))
        if num_edges < num_nodes - 1:
            raise ValueError("Le nombre d'arêtes doit être au moins égal à (n - 1) pour un graphe connexe.")
    except (ValueError, TypeError):
        messagebox.showerror("Erreur", "Entrées invalides. Veuillez entrer des entiers valides.")
        return

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))

    # Générer un arbre couvrant minimal pour garantir la connexité
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    for i in range(len(nodes) - 1):
        weight = random.randint(1, 20)
        graph.add_edge(nodes[i], nodes[i + 1], weight=weight)

    # Ajouter des arêtes supplémentaires aléatoires
    edges = set(graph.edges)
    while len(edges) < num_edges:
        u, v = random.sample(range(num_nodes), 2)
        weight = random.randint(1, 20)
        if u != v and (u, v) not in edges and (v, u) not in edges:
            edges.add((u, v))
            graph.add_edge(u, v, weight=weight)

    source = random.choice(list(graph.nodes))
    distances, paths = nx.single_source_dijkstra(graph, source=source)

    # Construire les résultats
    result = f"Distances minimales depuis le sommet source {source} :\n"
    for target, distance in distances.items():
        path = " -> ".join(map(str, paths[target]))
        result += f"Vers {target} : Distance = {distance}, Chemin = {path}\n"
    messagebox.showinfo("Résultats Dijkstra", result)

    # Visualisation du graphe
    pos = nx.spring_layout(graph, seed=42)
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    color_map = ["red" if node == source else "blue" for node in graph.nodes]

    plt.figure(figsize=(10, 8))
    nx.draw(
        graph, pos, with_labels=True, node_color=color_map,
        node_size=500, font_color='white', font_weight="bold"
    )
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.title(f"Graphe avec Dijkstra (Source : {source})")
    plt.show()


# Algorithme de Kruskal
# Algorithme de Kruskal
def kruskal_algorithm(graph):
    edges = sorted(graph.edges(data=True), key=lambda x: x[2]["weight"])
    mst = nx.Graph()
    mst.add_nodes_from(graph.nodes)

    subsets = {node: node for node in graph.nodes}

    def find(node):
        if subsets[node] != node:
            subsets[node] = find(subsets[node])
        return subsets[node]

    def union(node1, node2):
        root1 = find(node1)
        root2 = find(node2)
        if root1 != root2:
            subsets[root2] = root1

    mst_edges = []
    total_weight = 0

    for u, v, data in edges:
        if find(u) != find(v):
            union(u, v)
            mst.add_edge(u, v, weight=data["weight"])
            mst_edges.append((u, v, data["weight"]))
            total_weight += data["weight"]

    return mst, mst_edges, total_weight, edges


def execute_kruskal():
    try:
        num_nodes = int(simpledialog.askstring("Entrée", "Entrez le nombre de sommets du graphe :"))
        num_edges = int(simpledialog.askstring("Entrée", "Entrez le nombre d'arêtes du graphe :"))
        
        if num_edges < num_nodes - 1:
            raise ValueError("Le nombre d'arêtes doit être au moins égal à n-1 pour garantir un graphe connexe.")
    except (ValueError, TypeError):
        messagebox.showerror("Erreur", "Entrées invalides. Le nombre d'arêtes doit être ≥ n-1.")
        return

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))

    # Assurer la connexité en créant un arbre couvrant minimal
    nodes = list(range(num_nodes))
    random.shuffle(nodes)
    for i in range(1, num_nodes):
        u, v = nodes[i - 1], nodes[i]
        weight = random.randint(1, 20)
        graph.add_edge(u, v, weight=weight)

    # Ajouter des arêtes supplémentaires si nécessaire
    edges_to_add = num_edges - (num_nodes - 1)
    while edges_to_add > 0:
        u, v = random.sample(range(num_nodes), 2)
        if not graph.has_edge(u, v):
            weight = random.randint(1, 20)
            graph.add_edge(u, v, weight=weight)
            edges_to_add -= 1

    # Appliquer l'algorithme de Kruskal
    mst, mst_edges, total_weight, all_edges = kruskal_algorithm(graph)

    result = f"Arbre couvrant minimal (Total Weight: {total_weight}):\n"
    for u, v, weight in mst_edges:
        result += f"Arête ({u}, {v}) avec poids {weight}\n"
    messagebox.showinfo("Résultats Kruskal", result)

    pos = nx.spring_layout(graph)

    # Coloration des arêtes
    edge_colors = ["green" if (u, v) in mst.edges or (v, u) in mst.edges else "red" for u, v, _ in all_edges]

    # Dessiner le graphe complet
    plt.figure(figsize=(8, 6))
    nx.draw(graph, pos, with_labels=True, node_color="lightblue", node_size=500, font_weight="bold")
    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    # Dessiner les arêtes avec les couleurs appropriées
    nx.draw_networkx_edges(graph, pos, edgelist=all_edges, edge_color=edge_colors, width=2)
    
    # Dessiner l'arbre couvrant minimal en vert
    nx.draw(
        mst,
        pos,
        with_labels=True,
        node_color="lightgreen",
        node_size=500,
        font_weight="bold"
    )
    nx.draw_networkx_edge_labels(mst, pos, edge_labels=nx.get_edge_attributes(mst, "weight"))

    plt.title(f"Arbre couvrant minimal (Poids total : {total_weight})")
    plt.show()

# Algorithme de Bellman-Ford
def execute_bellmanford():
    try:
        num_nodes = int(simpledialog.askstring("Entrée", "Entrez le nombre de sommets du graphe :"))
        num_edges = int(simpledialog.askstring("Entrée", "Entrez le nombre d'arêtes du graphe :"))
        
        # Validation des entrées
        if num_edges < num_nodes - 1:
            raise ValueError("Le nombre d'arêtes doit être au moins égal à n-1 pour garantir un graphe connexe.")
    except (ValueError, TypeError):
        messagebox.showerror("Erreur", "Entrées invalides. Le nombre d'arêtes doit être ≥ n-1.")
        return

    graph = nx.DiGraph()  # Graphe orienté pour Bellman-Ford
    graph.add_nodes_from(range(num_nodes))

    # Assurer la connexité en ajoutant un chemin reliant tous les sommets
    nodes = list(range(num_nodes))
    random.shuffle(nodes)
    for i in range(1, num_nodes):
        u, v = nodes[i - 1], nodes[i]
        weight = random.randint(1, 20)
        graph.add_edge(u, v, weight=weight)

    # Ajouter des arêtes supplémentaires si nécessaire
    edges_to_add = num_edges - (num_nodes - 1)
    while edges_to_add > 0:
        u, v = random.sample(range(num_nodes), 2)
        if not graph.has_edge(u, v):
            weight = random.randint(1, 20)
            graph.add_edge(u, v, weight=weight)
            edges_to_add -= 1

    # Sélectionner une source aléatoire
    source = random.choice(list(graph.nodes))

    try:
        # Algorithme de Bellman-Ford
        distances, paths = nx.single_source_bellman_ford(graph, source=source)

        # Préparer les résultats
        result = f"Distances minimales depuis le sommet source {source} :\n"
        for target, distance in distances.items():
            path = " -> ".join(map(str, paths[target]))
            result += f"Vers {target} : Distance = {distance}, Chemin = {path}\n"
        messagebox.showinfo("Résultats Bellman-Ford", result)

        # Dessiner le graphe
        pos = nx.spring_layout(graph)
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        color_map = ["red" if node == source else "blue" for node in graph.nodes]

        plt.figure(figsize=(8, 6))
        nx.draw(
            graph, pos, with_labels=True, node_color=color_map,
            node_size=500, font_color='white', font_weight="bold"
        )
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
        plt.title(f"Graphe avec Bellman-Ford (Source : {source})")
        plt.show()
    except nx.NetworkXUnbounded:
        messagebox.showerror("Erreur", "Le graphe contient un cycle de poids négatif.")

# Algorithme de Potentiel Métra
def potentiel_metra_algorithm_with_details(nodes, edges):
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(edges)

    start_times = {node: 0 for node in nodes}
    finish_times = {node: 0 for node in nodes}
    levels = {node: 0 for node in nodes}

    sorted_nodes = list(nx.topological_sort(graph))
    for node in sorted_nodes:
        for predecessor in graph.predecessors(node):
            start_times[node] = max(start_times[node], finish_times[predecessor])
        finish_times[node] = start_times[node] + max(
            (graph[predecessor][node]['weight'] for predecessor in graph.predecessors(node)),
            default=0
        )
        levels[node] = (
            max(levels.get(predecessor, 0) + 1 for predecessor in graph.predecessors(node))
            if list(graph.predecessors(node)) else 0
        )

    # Assigner les niveaux comme un attribut aux nœuds
    nx.set_node_attributes(graph, levels, name="level")

    return start_times, finish_times, levels, graph

# Génération aléatoire des prédécesseurs et successeurs
def generate_random_graph(num_nodes):
    nodes = list(range(1, num_nodes + 1))
    edges = set()

    # Assurer la connexité en créant une chaîne reliant tous les sommets
    for i in range(2, num_nodes + 1):
        predecessor = random.randint(1, i - 1)
        weight = random.randint(1, 10)
        edges.add((predecessor, i, weight))

    # Ajouter des arêtes supplémentaires pour rendre le graphe plus dense
    while len(edges) < num_nodes + random.randint(0, num_nodes):
        u, v = random.sample(nodes, 2)
        if u != v and (u, v) not in edges:
            weight = random.randint(1, 10)
            edges.add((u, v, weight))

    return nodes, list(edges)
# Afficher le graphe dans une nouvelle fenêtre
def show_graph(graph, levels):
    graph_window = tk.Toplevel(gui)
    graph_window.title("Diagramme Potentiel-Metra")
    graph_window.geometry("800x600")

    pos = nx.multipartite_layout(graph, subset_key="level")  # Utiliser 'level' comme attribut de nœud
    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw(graph, pos, with_labels=True, node_size=700, node_color="lightblue", font_size=10, ax=ax)
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels={(u, v): d["weight"] for u, v, d in graph.edges(data=True)}, font_size=8, ax=ax
    )
    ax.set_title("Diagramme Potentiel-Metra", fontsize=14)

    # Intégrer le graphique dans la fenêtre
    canvas = FigureCanvasTkAgg(fig, master=graph_window)
    canvas.draw()

    # Créer un widget Tkinter pour afficher le graphique
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Fonction pour exécuter Potentiel Métra
def execute_potentiel_metra():
    try:
        num_nodes = int(simpledialog.askstring("Entrée", "Entrez le nombre de tâches (sommets) :"))
    except (ValueError, TypeError):
        messagebox.showerror("Erreur", "Entrée invalide. Veuillez entrer un entier valide.")
        return

    nodes, edges = generate_random_graph(num_nodes)

    try:
        start_times, finish_times, levels, graph = potentiel_metra_algorithm_with_details(nodes, edges)
    except nx.NetworkXUnfeasible:
        messagebox.showerror("Erreur", "Le graphe contient un cycle. Vérifiez les données générées.")
        return

    # Fenêtre des résultats
    result_window = tk.Toplevel(gui)
    result_window.title("Résultats Potentiel Métra")
    result_window.geometry("500x600")

    result_text = tk.Text(result_window, wrap=tk.WORD, font=("Arial", 10))
    result_text.insert(tk.END, "Tableau des sommets avec prédécesseurs et successeurs :\n\n")

    for node in nodes:
        predecessors = list(graph.predecessors(node))
        successors = list(graph.successors(node))
        result_text.insert(
            tk.END,
            f"Tâche {node}: Prédécesseurs -> {predecessors}, Successeurs -> {successors}\n"
        )

    result_text.insert(tk.END, "\nTemps de début, de fin et niveaux des tâches :\n\n")
    for node in nodes:
        result_text.insert(
            tk.END,
            f"Tâche {node}: Début -> {start_times[node]}, Fin -> {finish_times[node]}, Niveau -> {levels[node]}\n"
        )

    result_text.pack(fill=tk.BOTH, expand=True)

    # Afficher le graphe après un délai de 5 secondes
    result_window.after(5000, lambda: show_graph(graph, levels))

#Algorithme Moindre cout
def moindre_cout_method(supply, demand, cost_matrix):
    rows, cols = len(supply), len(demand)
    allocation = np.zeros((rows, cols), dtype=int)
    total_cost = 0
    cost_details = np.zeros((rows, cols), dtype=int)

    # Liste des coûts triés
    cost_indices = sorted(
        [(i, j) for i in range(rows) for j in range(cols)],
        key=lambda x: cost_matrix[x[0]][x[1]]
    )

    # Allocation des ressources en fonction du moindre coût
    for i, j in cost_indices:
        if supply[i] > 0 and demand[j] > 0:
            qty = min(supply[i], demand[j])
            allocation[i, j] = qty
            cost_details[i, j] = qty * cost_matrix[i, j]
            supply[i] -= qty
            demand[j] -= qty
            total_cost += cost_details[i, j]

    return allocation, cost_details, total_cost

# Fonction pour exécuter le moindre coût et afficher les détails
def execute_moindre_cout():
    try:
        # Entrées utilisateur
        supply = simpledialog.askstring("Entrée", "Entrez les capacités des usines (séparées par des espaces) :")
        if not supply:
            raise ValueError("Les capacités des usines sont manquantes.")
        demand = simpledialog.askstring("Entrée", "Entrez les demandes (séparées par des espaces) :")
        if not demand:
            raise ValueError("Les demandes sont manquantes.")
        cost_matrix_str = simpledialog.askstring("Entrée", "Entrez la matrice des coûts (lignes séparées par des points-virgules, colonnes par des espaces) :")
        if not cost_matrix_str:
            raise ValueError("La matrice des coûts est manquante.")

        # Conversion des entrées
        supply = list(map(int, supply.split()))
        demand = list(map(int, demand.split()))
        cost_matrix = np.array([list(map(int, row.split())) for row in cost_matrix_str.split(";")])

        # Vérification des données
        if sum(supply) != sum(demand):
            raise ValueError("La somme des capacités doit être égale à la somme des demandes.")

        # Exécution de l'algorithme du moindre coût
        allocation, cost_details, total_cost = moindre_cout_method(supply.copy(), demand.copy(), cost_matrix)

        # Affichage des résultats
        afficher_resultats(allocation, cost_matrix, cost_details, total_cost, supply, demand)

    except ValueError as e:
        messagebox.showerror("Erreur d'entrée", str(e))
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur est survenue : {str(e)}")

# Fonction pour afficher les résultats
def afficher_resultats(allocation, cost_matrix, cost_details, total_cost, supply, demand):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Création du tableau des résultats
    table_data = [["" for _ in range(len(demand) + 1)] for _ in range(len(supply) + 2)]
    table_data[0][0] = "Coût"
    for j in range(len(demand)):
        table_data[0][j + 1] = f"D{j + 1}"

    for i in range(len(supply)):
        table_data[i + 1][0] = f"U{i + 1}"
        for j in range(len(demand)):
            table_data[i + 1][j + 1] = f"{allocation[i, j]} ({cost_matrix[i, j]}×{allocation[i, j]}={cost_details[i, j]})"

    table_data[-1][0] = "Total"
    table_data[-1][1] = f"Coût Total = {total_cost}"

    # Affiche le tableau
    ax.axis("tight")
    ax.axis("off")
    ax.table(cellText=table_data, loc="center", cellLoc="center")
    plt.title("Résultats de l'algorithme du Moindre Coût")
    plt.show()

# Algorithme de Nord-Ouest
def nord_ouest_method(supply, demand, cost_matrix):
    rows, cols = len(supply), len(demand)
    allocation = np.zeros((rows, cols), dtype=int)
    i, j = 0, 0

    while i < rows and j < cols:
        allocation[i, j] = min(supply[i], demand[j])
        supply[i] -= allocation[i, j]
        demand[j] -= allocation[i, j]

        if supply[i] == 0:
            i += 1
        elif demand[j] == 0:
            j += 1

    total_cost = np.sum(allocation * cost_matrix)
    return allocation, total_cost

def execute_nord_ouest():
    try:
        supply = simpledialog.askstring("Entrée", "Entrez les capacités des usines (séparées par des espaces) :")
        demand = simpledialog.askstring("Entrée", "Entrez les demandes (séparées par des espaces) :")
        cost_matrix_str = simpledialog.askstring("Entrée", "Entrez la matrice des coûts (lignes séparées par des points-virgules, colonnes par des espaces) :")

        supply = list(map(int, supply.split()))
        demand = list(map(int, demand.split()))
        cost_matrix = np.array([list(map(int, row.split())) for row in cost_matrix_str.split(";")])

        if sum(supply) != sum(demand):
            messagebox.showerror("Erreur", "La somme des capacités doit être égale à la somme des demandes.")
            return

        allocation, total_cost = nord_ouest_method(supply, demand, cost_matrix)

        # Affichage des résultats
        fig, ax = plt.subplots()
        ax.table(cellText=allocation, rowLabels=[f"Usine {i+1}" for i in range(len(allocation))],
                 colLabels=[f"Demande {j+1}" for j in range(len(allocation[0]))],
                 loc="center", cellLoc="center")
        ax.axis("off")
        plt.title(f"Allocation finale (Coût total = {total_cost})")
        plt.show()
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur est survenue : {str(e)}")

def stepping_stone_method(supply, demand, cost_matrix, initial_allocation):
    rows, cols = len(supply), len(demand)
    allocation = initial_allocation.copy()
    total_cost = np.sum(allocation * cost_matrix)

    while True:
        # Étape 1 : Calcul des potentiels u et v
        u = [None] * rows
        v = [None] * cols
        u[0] = 0  # Fixer un potentiel initial

        while None in u or None in v:
            for i in range(rows):
                for j in range(cols):
                    if allocation[i, j] > 0:  # Si la cellule est dans la base
                        if u[i] is not None and v[j] is None:
                            v[j] = cost_matrix[i, j] - u[i]
                        elif v[j] is not None and u[i] is None:
                            u[i] = cost_matrix[i, j] - v[j]

        # Étape 2 : Calcul des coûts d'amélioration
        improvement_costs = np.full((rows, cols), np.inf)
        for i in range(rows):
            for j in range(cols):
                if allocation[i, j] == 0:  # Cellule non allouée
                    # Calcul du coût d'amélioration via un cycle
                    cycle = find_cycle(allocation, (i, j))
                    if cycle:
                        improvement_costs[i, j] = calculate_cycle_cost(cycle, cost_matrix)

        # Étape 3 : Vérification des coûts d'amélioration
        min_cost = improvement_costs.min()
        if min_cost >= 0:
            break  # Si aucun coût d'amélioration négatif, on termine

        # Étape 4 : Mise à jour de la solution
        i, j = np.unravel_index(np.argmin(improvement_costs), improvement_costs.shape)
        cycle = find_cycle(allocation, (i, j))
        update_allocation(allocation, cycle)

        # Mise à jour du coût total
        total_cost = np.sum(allocation * cost_matrix)

    return allocation, total_cost


def find_cycle(allocation, start):
    """
    Trouve un cycle basé sur l'allocation actuelle.
    """
    rows, cols = allocation.shape
    visited = set()
    stack = [(start, [])]

    while stack:
        (x, y), path = stack.pop()
        if (x, y) in visited:
            continue
        visited.add((x, y))
        new_path = path + [(x, y)]

        if len(new_path) > 3 and new_path[0] == (x, y):
            return new_path[:-1]  # Cycle trouvé

        # Ajouter les voisins (horizontaux et verticaux)
        for i in range(rows):
            if allocation[i, y] > 0 and (i, y) != (x, y):
                stack.append(((i, y), new_path))
        for j in range(cols):
            if allocation[x, j] > 0 and (x, j) != (x, y):
                stack.append(((x, j), new_path))

    return None  # Aucun cycle trouvé


def calculate_cycle_cost(cycle, cost_matrix):
    """
    Calcule le coût d'amélioration pour un cycle donné.
    """
    total = 0
    sign = 1  # Alternance entre + et -
    for i, j in cycle:
        total += sign * cost_matrix[i, j]
        sign *= -1
    return total


def update_allocation(allocation, cycle):
    """
    Met à jour l'allocation basée sur le cycle trouvé.
    """
    quantities = [allocation[i, j] for i, j in cycle if allocation[i, j] > 0]
    delta = min(quantities)
    sign = 1
    for i, j in cycle:
        allocation[i, j] += sign * delta
        sign *= -1


def execute_stepping_stone():
    try:
        # Entrées utilisateur
        supply = simpledialog.askstring("Entrée", "Entrez les capacités des usines (séparées par des espaces) :")
        demand = simpledialog.askstring("Entrée", "Entrez les demandes (séparées par des espaces) :")
        cost_matrix_str = simpledialog.askstring("Entrée", "Entrez la matrice des coûts (lignes séparées par des points-virgules, colonnes par des espaces) :")

        supply = list(map(int, supply.split()))
        demand = list(map(int, demand.split()))
        cost_matrix = np.array([list(map(int, row.split())) for row in cost_matrix_str.split(";")])

        if sum(supply) != sum(demand):
            messagebox.showerror("Erreur", "La somme des capacités doit être égale à la somme des demandes.")
            return

        # Utiliser une solution initiale (par exemple Moindre Coût)
        initial_allocation, _, _ = moindre_cout_method(supply.copy(), demand.copy(), cost_matrix)

        # Appliquer l'algorithme Stepping Stone
        allocation, total_cost = stepping_stone_method(supply, demand, cost_matrix, initial_allocation)

        # Affichage des résultats
        afficher_resultats(allocation, cost_matrix, allocation * cost_matrix, total_cost, supply, demand)
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur est survenue : {str(e)}")

def ford_fulkerson_algorithm(graph, source, sink):
    residual_graph = graph.copy()
    for u, v, data in residual_graph.edges(data=True):
        data['flow'] = 0  # Initialiser le flux à 0

    max_flow = 0

    def bfs_find_path():
        visited = {node: False for node in residual_graph.nodes}
        parent = {}
        queue = [source]
        visited[source] = True

        while queue:
            current = queue.pop(0)
            for neighbor in residual_graph.neighbors(current):
                capacity = residual_graph[current][neighbor]['capacity']
                flow = residual_graph[current][neighbor]['flow']
                if not visited[neighbor] and capacity > flow:  # Si on trouve un chemin valide
                    parent[neighbor] = current
                    visited[neighbor] = True
                    queue.append(neighbor)
                    if neighbor == sink:
                        path = []
                        while neighbor in parent:
                            path.append((parent[neighbor], neighbor))
                            neighbor = parent[neighbor]
                        return path[::-1]  # Retourner le chemin dans le bon ordre
        return None

    while True:
        path = bfs_find_path()
        if not path:
            break

        # Trouver le goulot d'étranglement (capacité résiduelle minimale sur le chemin)
        path_flow = min(
            residual_graph[u][v]['capacity'] - residual_graph[u][v]['flow'] for u, v in path
        )

        # Mettre à jour les flux dans le graphe résiduel
        for u, v in path:
            residual_graph[u][v]['flow'] += path_flow
            if residual_graph.has_edge(v, u):
                residual_graph[v][u]['flow'] -= path_flow
            else:
                residual_graph.add_edge(v, u, capacity=0, flow=-path_flow)

        max_flow += path_flow

    return max_flow, residual_graph



def execute_ford_fulkerson():
    try:
        num_nodes = int(simpledialog.askstring("Entrée", "Entrez le nombre de sommets du graphe :"))
        num_edges = int(simpledialog.askstring("Entrée", "Entrez le nombre d'arêtes du graphe :"))
    except (ValueError, TypeError):
        messagebox.showerror("Erreur", "Entrées invalides. Veuillez entrer des entiers valides.")
        return

    # Création du graphe orienté
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))

    # Générer un graphe connexe en ajoutant des arêtes
    for i in range(num_nodes - 1):
        capacity = random.randint(5, 20)
        graph.add_edge(i, i + 1, capacity=capacity)

    # Ajouter des arêtes supplémentaires pour atteindre le nombre d'arêtes souhaité
    edges = set(graph.edges())
    while len(edges) < num_edges:
        u, v = random.sample(range(num_nodes), 2)
        if u != v and not graph.has_edge(u, v):
            capacity = random.randint(5, 20)
            graph.add_edge(u, v, capacity=capacity)
            edges.add((u, v))

    # Définir le sommet source et le puits
    source = 0
    sink = num_nodes - 1

    # Exécuter l'algorithme de Ford-Fulkerson pour calculer le flux maximal
    max_flow, residual_graph = ford_fulkerson_algorithm(graph, source, sink)

    # Afficher le résultat dans une boîte de message
    result = f"Flux maximal du sommet {source} au sommet {sink} : {max_flow}\n\n"
    for u, v, data in residual_graph.edges(data=True):
        result += f"Arête ({u} -> {v}): Flux = {data.get('flow', 0)}/{data['capacity']}\n"
    messagebox.showinfo("Résultats Ford-Fulkerson", result)

    # Visualisation du graphe
    pos = nx.spring_layout(graph)
    edge_labels = {
        (u, v): f"{data.get('flow', 0)}/{data['capacity']}" for u, v, data in residual_graph.edges(data=True)
    }

    plt.figure(figsize=(10, 8))
    nx.draw(graph, pos, with_labels=True, node_color="lightblue", node_size=700, font_weight="bold")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color="red")
    plt.title(f"Graphe Résiduel - Flux Maximal: {max_flow}")
    plt.show()







photo = None

# Fonction principale de l'interface utilisateur
def get_user_input(prompt, title="Entrée"):
    try:
        return int(simpledialog.askstring(title, prompt))
    except (ValueError, TypeError):
        messagebox.showerror("Erreur", "Entrées invalides. Veuillez entrer un entier valide.")
        return None

# Fenêtre principale
def open_main_window():
    global gui,photo 
    gui = tk.Tk()
    gui.title("Graphes et Algorithmes")
    gui.geometry("600x600")
    gui.configure(bg="#121212")  # Fond noir mat, intense et agressif

    # Charger l'image depuis le chemin spécifié
    image_path = r"C:\Users\Adamb\.spyder-py3\application_RO\logo.png"
    image = Image.open(image_path)

    # Redimensionner l'image (petite taille)
    image = image.resize((500, 200), Image.Resampling.LANCZOS)

    # Convertir l'image pour Tkinter
    photo = ImageTk.PhotoImage(image)

    # Ajouter l'image dans un label
    image_label = tk.Label(gui, image=photo, bg="#121212")
    image_label.pack(pady=20)

    # Titre de la fenêtre avec un style futuriste
    title_label = tk.Label(
        gui,
        text="Bienvenue dans l'application des algorithmes du RO",
        font=("Roboto", 22, "bold"),
        fg="#f1f1f1",  # Texte blanc lumineux
        bg="#121212"   # Fond sombre intense
    )
    title_label.pack(pady=50)

    # Frame contenant les boutons pour une structure plus propre
    button_frame = tk.Frame(gui, bg="#121212")
    button_frame.pack(pady=30)

    # Bouton pour ouvrir le menu des algorithmes avec un design modifié
    open_button = tk.Button(
        button_frame,
        text="Ouvrir le Menu des Algorithmes",
        command=open_algo_menu,
        font=("Roboto", 16, "bold"),
        bg="#1e1e1e",  # Fond gris sombre
        fg="#f1f1f1",   # Texte lumineux
        activebackground="#ff4444",  # Rouge vif pour un effet percutant
        activeforeground="black",    # Texte noir quand survolé
        width=25,
        height=2,
        bd=3,
        relief="flat",  # Bordure plate et moderne
        cursor="hand2"
    )
    open_button.grid(row=0, column=0, padx=10, pady=10)

    # Bouton pour quitter l'application avec un look dynamique
    exit_button = tk.Button(
        button_frame,
        text="Quitter",
        command=gui.quit,
        font=("Roboto", 16, "bold"),
        bg="#d32f2f",  # Rouge vif agressif
        fg="#f1f1f1",   # Texte blanc lumineux
        activebackground="#b71c1c",  # Rouge foncé sur survol
        activeforeground="white",
        width=25,
        height=2,
        bd=3,
        relief="flat",  # Bordure plate
        cursor="hand2"
    )
    exit_button.grid(row=1, column=0, padx=10, pady=10)

    # Bouton d'information avec un look plus vibrant
    info_button = tk.Button(
        button_frame,
        text="Informations sur l'application",
        command=lambda: messagebox.showinfo("Informations sur l'application", "Visualisation et exécution d'algorithmes sur graphes\n\nCette application a été réalisée par Adam Bouzid et Nizar Afraoui\n\n et encadré par El mkhalet mouna"),
        font=("Roboto", 16, "bold"),
        bg="#388e3c",  # Vert foncé, lumineux et énergique
        fg="#f1f1f1",   # Texte blanc lumineux
        activebackground="#2c6f2c",  # Vert plus foncé au survol
        activeforeground="white",
        width=25,
        height=2,
        bd=3,
        relief="flat",  # Bordure plate et moderne
        cursor="hand2"
    )
    info_button.grid(row=2, column=0, padx=10, pady=10)

    # Lancer la boucle principale de l'interface
    gui.mainloop()


# Lancer la fenêtre principale
open_main_window()
