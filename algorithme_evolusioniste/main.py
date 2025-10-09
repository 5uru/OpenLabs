import jax
import jax.numpy as jnp
from jax import random
import string

# -----------------------------
# Configuration
# -----------------------------
TARGET = "Jonathan Suru"  # ← Remplacez par votre nom
POP_SIZE = 200          # Taille de la population
MUTATION_RATE = 0.05    # Probabilité de mutation par caractère
SEED = 42

# Encodage : convertir la chaîne cible en tableau d'entiers (ASCII)
target_chars = list(TARGET)
target_array = jnp.array([ord(c) for c in target_chars])
GENOME_LENGTH = len(target_array)

# Ensemble des caractères autorisés (lettres, espaces, ponctuation courante)
CHARSET = string.ascii_letters + string.digits + " .,-_"
CHARSET_ARRAY = jnp.array([ord(c) for c in CHARSET])
CHARSET_SIZE = len(CHARSET_ARRAY)

# -----------------------------
# Fonctions de l'algorithme
# -----------------------------

def create_random_genome(key):
    """Crée un génome aléatoire (tableau d'entiers ASCII)."""
    indices = random.randint(key, (GENOME_LENGTH,), minval=0, maxval=CHARSET_SIZE)
    return CHARSET_ARRAY[indices]

def create_population(key, pop_size):
    """Crée une population de génomes aléatoires."""
    keys = random.split(key, pop_size)
    return jax.vmap(create_random_genome)(keys)

def fitness(genome):
    """Fitness = nombre de caractères corrects à la bonne position."""
    return jnp.sum(genome == target_array)

def select_parent(population, fitnesses, key):
    """Sélection par tournoi (taille=3)."""
    idxs = random.randint(key, (3,), minval=0, maxval=POP_SIZE)
    chosen_fitnesses = fitnesses[idxs]
    best_idx = idxs[jnp.argmax(chosen_fitnesses)]
    return population[best_idx]

def mutate(genome, key):
    """Mutation : chaque caractère a une probabilité MUTATION_RATE d'être changé."""
    mutate_mask = random.uniform(key, (GENOME_LENGTH,)) < MUTATION_RATE
    new_chars_idx = random.randint(key, (GENOME_LENGTH,), minval=0, maxval=CHARSET_SIZE)
    new_chars = CHARSET_ARRAY[new_chars_idx]
    return jnp.where(mutate_mask, new_chars, genome)

def crossover(parent1, parent2, key):
    """Croisement à un point aléatoire."""
    point = random.randint(key, (), minval=0, maxval=GENOME_LENGTH)
    child = jnp.concatenate([parent1[:point], parent2[point:]])
    return child

def decode_genome(genome):
    """Convertit un génome (entiers) en chaîne de caractères."""
    return ''.join(chr(int(x)) for x in genome)

# -----------------------------
# Boucle principale
# -----------------------------

def main():
    key = random.PRNGKey(SEED)

    # Initialisation
    key, subkey = random.split(key)
    population = create_population(subkey, POP_SIZE)

    generation = 0
    best_fitness = 0

    while True:
        # Calcul des fitness
        fitnesses = jax.vmap(fitness)(population)
        best_idx = jnp.argmax(fitnesses)
        best_fitness = int(fitnesses[best_idx])
        best_individual = population[best_idx]

        # Affichage
        current_str = decode_genome(best_individual)
        print(f"Génération {generation:4d} | Meilleur: '{current_str}' | Fitness: {best_fitness}/{GENOME_LENGTH}")

        # Arrêt si trouvé
        if best_fitness == GENOME_LENGTH:
            print("\n✅ Cible atteinte !")
            break

        # Nouvelle génération
        new_population = []
        for i in range(POP_SIZE):
            key, k1, k2, k3, k4 = random.split(key, 5)
            parent1 = select_parent(population, fitnesses, k1)
            parent2 = select_parent(population, fitnesses, k2)
            child = crossover(parent1, parent2, k3)
            child = mutate(child, k4)
            new_population.append(child)

        population = jnp.stack(new_population)
        generation += 1

        # Sécurité : arrêt après 5000 générations
        if generation > 5000:
            print("\n❌ Échec : limite de générations atteinte.")
            break

if __name__ == "__main__":
    main()