import math
import random

from PIL import Image, ImageDraw


def generate_unique_wax_pattern(size=200, seed=None):
    """
    Generate a unique, random wax pattern.
    Returns a PIL Image object.
    """
    if seed is not None:
        random.seed(seed)

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    margin = 10
    center = size // 2
    max_radius = size // 2 - margin

    palettes = [
        [(139, 0, 0), (255, 215, 0), (0, 100, 0)],
        [(0, 0, 139), (255, 255, 0), (255, 140, 0)],
        [(128, 0, 128), (255, 192, 203), (0, 191, 255)],
        [(178, 34, 34), (240, 230, 140), (34, 139, 34)],
        [(255, 69, 0), (255, 255, 224), (0, 100, 100)],
        [(75, 0, 130), (255, 20, 147), (255, 255, 0)],
        [(0, 128, 128), (255, 165, 0), (255, 255, 240)],
        [(107, 142, 35), (0, 0, 0), (255, 215, 0)],
        [(220, 20, 60), (255, 255, 0), (0, 128, 0)],
        [(139, 0, 139), (0, 255, 255), (255, 105, 180)],
        [(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        [(255, 127, 80), (60, 179, 113), (70, 130, 180)],
        [(255, 99, 71), (255, 255, 54), (100, 149, 237)],
        [(205, 92, 92), (238, 232, 170), (102, 205, 170)],
        [(250, 128, 114), (245, 245, 220), (176, 224, 230)],
    ]
    palette = random.choice(palettes)

    def get_color():
        return random.choice(palette)

    def pattern_01():  # 8-pointed star
        color = get_color()
        points = []
        for i in range(8):
            angle = math.radians(i * 45 - 90)
            radius = max_radius if i % 2 == 0 else max_radius // 2
            x = center + radius * math.cos(angle)
            y = center + radius * math.sin(angle)
            points.append((x, y))
        for j in range(4):
            p1 = points[j * 2]
            p2 = points[(j * 2 + 3) % 8]
            p3 = points[(j * 2 + 1) % 8]
            draw.polygon([p1, p2, p3], fill=color)

    def pattern_02():  # Concentric circles
        for i in range(3, 0, -1):
            r = max_radius * i // 3
            draw.ellipse(
                [center - r, center - r, center + r, center + r], fill=get_color()
            )

    def pattern_03():  # Diamond
        color = get_color()
        points = [
            (center, center - max_radius),
            (center + max_radius, center),
            (center, center + max_radius),
            (center - max_radius, center),
        ]
        draw.polygon(points, fill=color)

    def pattern_04():  # Triangle with random rotation
        color = get_color()
        rotation = random.choice([0, 1, 2, 3])
        if rotation == 0:
            points = [
                (center, center - max_radius),
                (center + max_radius, center + max_radius),
                (center - max_radius, center + max_radius),
            ]
        elif rotation == 1:
            points = [
                (center + max_radius, center),
                (center - max_radius, center + max_radius),
                (center - max_radius, center - max_radius),
            ]
        elif rotation == 2:
            points = [
                (center, center + max_radius),
                (center + max_radius, center - max_radius),
                (center - max_radius, center - max_radius),
            ]
        else:
            points = [
                (center - max_radius, center),
                (center + max_radius, center + max_radius),
                (center + max_radius, center - max_radius),
            ]
        draw.polygon(points, fill=color)

    def pattern_05():  # Hexagon
        color = get_color()
        points = []
        for i in range(6):
            angle = math.radians(i * 60 + random.randint(0, 30))
            x = center + max_radius * math.cos(angle)
            y = center + max_radius * math.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=color)

    def pattern_06():  # Rotated square
        color = get_color()
        points = [
            (center, center - max_radius),
            (center + max_radius, center),
            (center, center + max_radius),
            (center - max_radius, center),
        ]
        draw.polygon(points, fill=color)

    def pattern_07():  # Cross
        color = get_color()
        thickness = random.randint(max_radius // 4, max_radius // 2)
        draw.rectangle(
            [
                center - thickness,
                center - max_radius,
                center + thickness,
                center + max_radius,
            ],
            fill=color,
        )
        draw.rectangle(
            [
                center - max_radius,
                center - thickness,
                center + max_radius,
                center + thickness,
            ],
            fill=color,
        )

    def pattern_08():  # Flower
        num_petals = random.randint(4, 8)
        color = get_color()
        for i in range(num_petals):
            angle = math.radians(i * 360 / num_petals)
            px = center + max_radius // 2 * math.cos(angle)
            py = center + max_radius // 2 * math.sin(angle)
            draw.ellipse(
                [
                    px - max_radius // 2,
                    py - max_radius // 2,
                    px + max_radius // 2,
                    py + max_radius // 2,
                ],
                fill=color,
            )
        draw.ellipse(
            [
                center - max_radius // 3,
                center - max_radius // 3,
                center + max_radius // 3,
                center + max_radius // 3,
            ],
            fill=get_color(),
        )

    def pattern_09():  # Sun
        color = get_color()
        num_rays = random.randint(8, 16)
        for i in range(num_rays):
            angle = math.radians(i * 360 / num_rays)
            x1 = center + max_radius // 3 * math.cos(angle)
            y1 = center + max_radius // 3 * math.sin(angle)
            x2 = center + max_radius * math.cos(angle)
            y2 = center + max_radius * math.sin(angle)
            draw.line([(x1, y1), (x2, y2)], fill=color, width=random.randint(2, 5))
        draw.ellipse(
            [
                center - max_radius // 3,
                center - max_radius // 3,
                center + max_radius // 3,
                center + max_radius // 3,
            ],
            fill=color,
        )

    def pattern_10():  # Wave
        color = get_color()
        points = []
        for i in range(15):
            x = center - max_radius + i * max_radius * 2 // 15
            y = center + max_radius // 2 * math.sin(i * math.pi / 3)
            points.append((x, y))
        draw.line(points, fill=color, width=random.randint(5, 15))

    def pattern_11():  # Spiral
        color = get_color()
        points = []
        for i in range(25):
            angle = math.radians(i * 45)
            radius = i * max_radius // 25
            x = center + radius * math.cos(angle)
            y = center + radius * math.sin(angle)
            points.append((x, y))
        if len(points) > 2:
            draw.line(points, fill=color, width=random.randint(2, 5))

    def pattern_12():  # Checkerboard
        colors = [get_color(), get_color()]
        for i in range(4):
            for j in range(4):
                x = center - max_radius + i * max_radius // 2
                y = center - max_radius + j * max_radius // 2
                c = colors[(i + j) % 2]
                draw.rectangle([x, y, x + max_radius // 2, y + max_radius // 2], fill=c)

    def pattern_13():  # Chevron
        for i in range(3):
            y_offset = i * max_radius // 2 - max_radius // 2
            color = get_color()
            points = [
                (center - max_radius, center + y_offset),
                (center, center + y_offset + max_radius // 2),
                (center + max_radius, center + y_offset),
                (center + max_radius, center + y_offset - max_radius // 4),
                (center, center + y_offset + max_radius // 4),
                (center - max_radius, center + y_offset - max_radius // 4),
            ]
            draw.polygon(points, fill=color)

    def pattern_14():  # Stripes (diagonal, horizontal, or vertical)
        direction = random.choice(["diag", "horiz", "vert"])
        for i in range(-max_radius, max_radius * 2, 20):
            color = get_color()
            if direction == "diag":
                x1, y1 = center - max_radius + i, center - max_radius
                x2, y2 = center + max_radius + i, center + max_radius
            elif direction == "horiz":
                x1, y1 = center - max_radius, center - max_radius + i
                x2, y2 = center + max_radius, center - max_radius + i
            else:
                x1, y1 = center - max_radius + i, center - max_radius
                x2, y2 = center - max_radius + i, center + max_radius
            draw.line([(x1, y1), (x2, y2)], fill=color, width=random.randint(5, 12))

    def pattern_16():  # Zigzag
        color = get_color()
        points = [(center - max_radius, center)]
        for i in range(10):
            x = center - max_radius + i * max_radius // 5
            y = center + max_radius // 2 if i % 2 == 0 else center - max_radius // 2
            points.append((x, y))
        draw.line(points, fill=color, width=random.randint(5, 12))

    def pattern_17():  # Pentagon
        color = get_color()
        points = []
        for i in range(5):
            angle = math.radians(i * 72 - 90 + random.randint(0, 30))
            x = center + max_radius * math.cos(angle)
            y = center + max_radius * math.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=color)

    def pattern_18():  # Octagon
        color = get_color()
        points = []
        for i in range(8):
            angle = math.radians(i * 45 + random.randint(0, 20))
            x = center + max_radius * math.cos(angle)
            y = center + max_radius * math.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=color)

    def pattern_19():  # African style 1
        color = get_color()
        draw.ellipse(
            [
                center - max_radius // 3,
                center - max_radius // 3,
                center + max_radius // 3,
                center + max_radius // 3,
            ],
            fill=color,
        )
        for i in range(4):
            angle = math.radians(i * 90)
            px = center + max_radius // 2 * math.cos(angle)
            py = center + max_radius // 2 * math.sin(angle)
            points = [(px, py), (px + 30, py + 30), (px - 30, py + 30)]
            draw.polygon(points, fill=get_color())

    def pattern_20():  # African style 2 - layered diamonds
        for i in range(3):
            t = max_radius - i * max_radius // 3
            color = get_color()
            points = [
                (center, center - t),
                (center + t, center),
                (center, center + t),
                (center - t, center),
            ]
            draw.polygon(points, fill=color)

    def pattern_21():  # African style 3 - rings with dots
        color = get_color()
        draw.ellipse(
            [
                center - max_radius,
                center - max_radius,
                center + max_radius,
                center + max_radius,
            ],
            outline=color,
            width=3,
        )
        draw.ellipse(
            [
                center - max_radius // 2,
                center - max_radius // 2,
                center + max_radius // 2,
                center + max_radius // 2,
            ],
            fill=color,
        )
        for i in range(4):
            angle = math.radians(i * 90)
            px = center + max_radius // 1.5 * math.cos(angle)
            py = center + max_radius // 1.5 * math.sin(angle)
            draw.ellipse([px - 8, py - 8, px + 8, py + 8], fill=get_color())

    def pattern_22():  # Arrow
        color = get_color()
        direction = random.choice(["up", "down", "left", "right"])
        if direction == "up":
            points = [
                (center, center - max_radius - 10),
                (center - 20, center),
                (center + 20, center),
            ]
        elif direction == "down":
            points = [
                (center, center + max_radius + 10),
                (center - 20, center),
                (center + 20, center),
            ]
        elif direction == "left":
            points = [
                (center - max_radius - 10, center),
                (center, center - 20),
                (center, center + 20),
            ]
        else:
            points = [
                (center + max_radius + 10, center),
                (center, center - 20),
                (center, center + 20),
            ]
        draw.polygon(points, fill=color)

    def pattern_23():  # Knot / infinity symbol
        color = get_color()
        draw.ellipse(
            [
                center - max_radius,
                center - max_radius // 2,
                center,
                center + max_radius // 2,
            ],
            fill=color,
        )
        draw.ellipse(
            [
                center,
                center - max_radius // 2,
                center + max_radius,
                center + max_radius // 2,
            ],
            fill=color,
        )
        draw.ellipse(
            [
                center - max_radius // 2,
                center - max_radius,
                center + max_radius // 2,
                center,
            ],
            fill=get_color(),
        )

    def pattern_24():  # Lightning bolt
        color = get_color()
        points = [
            (center - 10, center - max_radius),
            (center + 20, center - max_radius // 3),
            (center - 10, center - max_radius // 3),
            (center + 20, center + max_radius),
            (center - 30, center),
            (center, center),
        ]
        draw.polygon(points, fill=color)

    def pattern_25():  # Grid
        color = get_color()
        for i in range(5):
            x = center - max_radius + i * max_radius // 2
            draw.line(
                [(x, center - max_radius), (x, center + max_radius)],
                fill=color,
                width=2,
            )
            y = center - max_radius + i * max_radius // 2
            draw.line(
                [(center - max_radius, y), (center + max_radius, y)],
                fill=color,
                width=2,
            )

    def pattern_26():  # Interlaced ellipses
        color = get_color()
        draw.ellipse(
            [
                center - max_radius,
                center - max_radius // 2,
                center + max_radius,
                center + max_radius // 2,
            ],
            fill=color,
        )
        draw.ellipse(
            [
                center - max_radius // 2,
                center - max_radius,
                center + max_radius // 2,
                center + max_radius,
            ],
            fill=get_color(),
        )

    def pattern_27():  # Shell / spiral arcs
        color = get_color()
        for i in range(8):
            angle = math.radians(i * 45)
            radius = max_radius - i * max_radius // 8
            draw.arc(
                [center - radius, center - radius, center + radius, center + radius],
                math.degrees(angle) - 10,
                math.degrees(angle) + 10,
                fill=color,
                width=5,
            )

    def pattern_28():  # 12-pointed star
        color = get_color()
        points = []
        for i in range(12):
            angle = math.radians(i * 30 - 90)
            radius = max_radius if i % 2 == 0 else max_radius // 2
            x = center + radius * math.cos(angle)
            y = center + radius * math.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=color)

    all_patterns = [
        pattern_01,
        pattern_02,
        pattern_03,
        pattern_04,
        pattern_05,
        pattern_06,
        pattern_07,
        pattern_08,
        pattern_09,
        pattern_10,
        pattern_11,
        pattern_12,
        pattern_13,
        pattern_14,
        pattern_16,
        pattern_17,
        pattern_18,
        pattern_19,
        pattern_20,
        pattern_21,
        pattern_22,
        pattern_23,
        pattern_24,
        pattern_25,
        pattern_26,
        pattern_27,
        pattern_28,
    ]

    chosen_pattern = random.choice(all_patterns)
    chosen_pattern()
    return img


def generate_wax_patterns(count=10, size=28, seed=None):
    """
    Generate a list of unique wax patterns as PIL Image objects.

    Args:
        count (int): Number of patterns to generate.
        size (int): Size of each pattern in pixels.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        list: List of PIL Image objects.
    """
    if seed is not None:
        random.seed(seed)

    patterns = []
    for i in range(count):
        pattern_seed = seed + i if seed is not None else None
        img = generate_unique_wax_pattern(size=size, seed=pattern_seed)
        patterns.append(img)

    return patterns
