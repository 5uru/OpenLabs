import numpy as np
import matplotlib.pyplot as plt

def generate_wax_cell_28x28(seed=None):
    size = 28
    if seed is not None:
        np.random.seed(seed)

    palette = np.array([
            [1.0, 0.0, 0.0],   # Rouge
            [0.0, 0.0, 1.0],   # Bleu
            [1.0, 1.0, 0.0],   # Jaune
            [0.0, 1.0, 0.0],   # Vert
            [1.0, 0.5, 0.0],   # Orange
            [0.8, 0.0, 0.8],   # Violet
            [0.0, 0.0, 0.0],   # Noir
            [1.0, 1.0, 1.0],   # Blanc
    ])

    bg_color = palette[np.random.choice([6, 7])]
    cell = np.tile(bg_color, (size, size, 1))

    # Ajout du motif 'star'
    motif_type = np.random.choice([ 'star']) # 'geometric', 'floral', 'leaf',
    fg_color = palette[np.random.randint(0, 6)]
    contour = palette[6]

    y, x = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    center = (size // 2, size // 2)
    dx = x - center[1]
    dy = y - center[0]
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    if motif_type == 'geometric':
        # Choisir aléatoirement une ou plusieurs formes géométriques
        shapes = np.random.choice(['diamond', 'circle', 'triangle', 'hexagon', 'cross', 'ring'],
                                  size=np.random.randint(1, 4), replace=False)

        for shape in shapes:
            if shape == 'diamond':
                mask = (np.abs(dx) + np.abs(dy)) <= 8
                cell[mask] = fg_color
            elif shape == 'circle':
                mask = r <= 7
                cell[mask] = fg_color
            elif shape == 'triangle':
                # Triangle pointé vers le haut
                height = 10
                base = 16
                tri_mask = (dy >= -height) & (dy <= 0) & (np.abs(dx) <= (base/2) * (1 + dy / height))
                cell[tri_mask] = fg_color
            elif shape == 'hexagon':
                # Hexagone approximé avec 6 inégalités
                a = 8
                hex_mask = (
                        (dx >= -a) & (dx <= a) &
                        (dy >= -a * np.sqrt(3)/2) & (dy <= a * np.sqrt(3)/2) &
                        (dy <= np.sqrt(3) * dx + a * np.sqrt(3)) &
                        (dy >= -np.sqrt(3) * dx - a * np.sqrt(3)) &
                        (dy <= -np.sqrt(3) * dx + a * np.sqrt(3)) &
                        (dy >= np.sqrt(3) * dx - a * np.sqrt(3))
                )
                cell[hex_mask] = fg_color
            elif shape == 'cross':
                width = 3
                cross_mask = (np.abs(dx) <= width) | (np.abs(dy) <= width)
                cell[cross_mask] = fg_color
            elif shape == 'ring':
                inner, outer = 4, 8
                ring_mask = (r >= inner) & (r <= outer)
                cell[ring_mask] = fg_color

        # Petit cercle central avec couleur aléatoire
        small_circle = r <= 3
        cell[small_circle] = palette[np.random.randint(0, 6)]

    elif motif_type == 'floral':
        n_petals = np.random.choice([4, 5, 6])
        for k in range(n_petals):
            angle = 2 * np.pi * k / n_petals
            petal = (r <= 9) & (np.abs((theta - angle + np.pi) % (2*np.pi) - np.pi) < 0.6)
            curve = np.sin(4 * (theta - angle)) * 0.3 + 0.6
            line_mask = petal & (r <= 9 * curve)
            cell[line_mask] = fg_color

    elif motif_type == 'leaf':
        a, b = 10, 5
        cos_t, sin_t = np.cos(np.pi/4), np.sin(np.pi/4)
        dx_rot = dx * cos_t - dy * sin_t
        dy_rot = dx * sin_t + dy * cos_t
        leaf = (dx_rot**2 / a**2 + dy_rot**2 / b**2) <= 1
        cell[leaf] = fg_color
        vein = (np.abs(dy_rot) < 1.5) & (dx_rot > -a) & (dx_rot < a)
        cell[vein] = contour

    elif motif_type == 'star':
        # Étoile à 5 branches
        n = 5
        outer_r = 10
        inner_r = 4
        angles = np.linspace(0, 2*np.pi, 2*n, endpoint=False)
        radii = np.array([outer_r if i % 2 == 0 else inner_r for i in range(2*n)])
        star_mask = np.zeros_like(r, dtype=bool)
        for i in range(28):
            for j in range(28):
                pt_angle = np.arctan2(dy[i, j], dx[i, j])
                pt_r = r[i, j]
                # Trouver l'angle le plus proche
                diff = np.abs((angles - pt_angle + np.pi) % (2*np.pi) - np.pi)
                closest_idx = np.argmin(diff)
                max_r_at_angle = np.interp(pt_angle, angles, radii, period=2*np.pi)
                if pt_r <= max_r_at_angle:
                    star_mask[i, j] = True
        cell[star_mask] = fg_color

    # Toujours un point central coloré
    center_dot = r <= 2
    cell[center_dot] = palette[np.random.randint(0, 6)]

    return cell, bg_color

def has_visible_motif(cell, bg_color, min_diff_ratio=0.05):
    diff = np.any(np.abs(cell - bg_color) > 0.01, axis=2)
    return np.mean(diff) >= min_diff_ratio
def main(n=100):
    cells = []
    seed = 2025
    attempts = 0

    while len(cells) < n and attempts < 300:
        cell, bg = generate_wax_cell_28x28(seed=seed)
        if has_visible_motif(cell, bg, min_diff_ratio=0.05):
            cells.append(cell)
        seed += 1
        attempts += 1
    cells = np.array(cells)  # (100, 28, 28, 3)
    return cells


