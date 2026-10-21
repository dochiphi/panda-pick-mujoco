import numpy as np
from PIL import Image

img = np.array(Image.open("top_view.png"))
print("Image shape:", img.shape)
for row, col in [(196, 320), (190, 320), (200, 320), (196, 310), (196, 330), (240, 320)]:
    print(f"row {row}, col {col}: RGB = {img[row, col][:3]}")
