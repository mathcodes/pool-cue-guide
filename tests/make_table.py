"""Synthetic table photo for tests: felt, rails, pockets, 9 balls, mild perspective."""
import cv2, numpy as np

def make(path="/tmp/table.jpg", seed=3):
    rng = np.random.default_rng(seed)
    W, H = 1400, 800
    img = np.full((H, W, 3), (60, 70, 85), np.uint8)               # floor
    cv2.rectangle(img, (120, 90), (1280, 710), (30, 60, 110), -1)    # rail (wood, BGR)
    cv2.rectangle(img, (170, 140), (1230, 660), (70, 140, 30), -1)   # felt
    for x in (170, 700, 1230):
        for y in (140, 660):
            cv2.circle(img, (x, y), 28, (10, 10, 10), -1)           # pockets
    balls = [  # x, y, bgr, stripe
        (420, 400, (245, 245, 245), False),   # cue
        (760, 330, (15, 15, 15), False),      # eight
        (880, 420, (30, 30, 200), False), (960, 360, (40, 160, 230), False),
        (1000, 470, (200, 60, 40), False), (820, 520, (30, 30, 200), True),
        (640, 250, (40, 160, 230), True), (560, 540, (200, 60, 40), True),
        (1050, 270, (30, 120, 30), False),
    ]
    for x, y, c, stripe in balls:
        cv2.circle(img, (x + 5, y + 6), 24, (40, 100, 20), -1)      # soft shadow
        cv2.circle(img, (x, y), 24, c, -1, cv2.LINE_AA)
        if stripe:
            cv2.rectangle(img, (x - 24, y - 7), (x + 24, y + 7), (245, 245, 245), -1)
            cv2.circle(img, (x, y), 24, c, 2, cv2.LINE_AA)
        cv2.circle(img, (x - 8, y - 9), 5, (255, 255, 255), -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = rng.normal(0, 3, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)
    return path

if __name__ == "__main__":
    print(make())
