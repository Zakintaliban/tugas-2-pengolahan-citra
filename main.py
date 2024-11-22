import imageio
import numpy as np
import matplotlib.pyplot as plt

# Membaca gambar berwarna (RGB)
image = imageio.imread("./gambar.jpg")

# Mengubah gambar menjadi grayscale
gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

# Menampilkan output dari gray_image
imageio.imwrite("./gray_image.jpg", gray_image)

# Menghitung histogram
total_pixels_per_intensity = np.zeros(256, dtype=int)
for intensity in range(256):
    total_pixels_per_intensity[intensity] = np.sum(gray_image == intensity)

# Plot histogram
plt.figure(figsize=(10, 6))
plt.bar(range(256), total_pixels_per_intensity, color="gray")
plt.xlabel("Intensitas Piksel")
plt.ylabel("Jumlah Piksel")
plt.title("Histogram Gambar Grayscale")
plt.show()

# Pertanyaan a: Jumlah total piksel untuk setiap intensitas pada gambar grayscale
for intensity, count in enumerate(total_pixels_per_intensity):
    print(f"Intensitas {intensity}: {count} piksel")

# Menambahkan total jumlah piksel
total_pixels = np.sum(total_pixels_per_intensity)
print(f"Jumlah total piksel: {total_pixels} piksel")

# Pertanyaan b: Apakah ada intensitas tertentu yang dominan dalam gambar tersebut?
dominant_intensity = np.argmax(total_pixels_per_intensity)
print(
    f"Intensitas dominan adalah {dominant_intensity} dengan jumlah {total_pixels_per_intensity[dominant_intensity]} piksel."
)
