import cv2
import matplotlib.pyplot as plt
import os

# ============================================
# MEMBUAT FOLDER HASIL
# ============================================

os.makedirs("hasil/grayscale", exist_ok=True)
os.makedirs("hasil/histogram", exist_ok=True)


# ============================================
# DAFTAR NAMA FILE GAMBAR
# ============================================

nama_file = [
    "gamba.1.jpg",
    "gambar.2.jpg",
    "gambar.3..jpg",
    "gambar.4.jpg",
    "gambar.5.jpg",
    "gambar.6.jpg",
    "gambar.7.jpg",
    "gambar.8.jpg",
    "gambar.9.jpg"
]


# ============================================
# PROSES GAMBAR
# ============================================

for i, file in enumerate(nama_file, start=1):

    print(f"Memproses citra {i}: {file}")

    lokasi_file = os.path.join("gambar", file)

    # Membaca gambar
    citra = cv2.imread(lokasi_file)

    if citra is None:
        print(f"File tidak ditemukan: {file}")
        continue

    # BGR ke RGB
    rgb = cv2.cvtColor(citra, cv2.COLOR_BGR2RGB)

    # RGB ke Grayscale
    grayscale = cv2.cvtColor(citra, cv2.COLOR_BGR2GRAY)


    # ========================================
    # SIMPAN GRAYSCALE
    # ========================================

    nama_grayscale = f"hasil/grayscale/grayscale_{i}.jpg"

    cv2.imwrite(nama_grayscale, grayscale)

    print(f"Grayscale tersimpan: {nama_grayscale}")


    # ========================================
    # MEMBUAT HISTOGRAM
    # ========================================

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))


    # Citra RGB
    ax[0, 0].imshow(rgb)
    ax[0, 0].set_title(f"Citra RGB - Citra {i}")
    ax[0, 0].axis("off")


    # Histogram RGB
    warna = ["red", "green", "blue"]

    for j, warna_channel in enumerate(warna):

        histogram = cv2.calcHist(
            [rgb],
            [j],
            None,
            [256],
            [0, 256]
        )

        ax[0, 1].plot(
            histogram,
            color=warna_channel,
            label=warna_channel.upper()
        )

    ax[0, 1].set_title(f"Histogram RGB - Citra {i}")
    ax[0, 1].set_xlabel("Intensitas Piksel")
    ax[0, 1].set_ylabel("Jumlah Piksel")
    ax[0, 1].set_xlim([0, 256])
    ax[0, 1].legend()


    # Citra Grayscale
    ax[1, 0].imshow(grayscale, cmap="gray")
    ax[1, 0].set_title(f"Citra Grayscale - Citra {i}")
    ax[1, 0].axis("off")


    # Histogram Grayscale
    ax[1, 1].hist(
        grayscale.ravel(),
        bins=256,
        range=(0, 256)
    )

    ax[1, 1].set_title(f"Histogram Grayscale - Citra {i}")
    ax[1, 1].set_xlabel("Intensitas Piksel")
    ax[1, 1].set_ylabel("Jumlah Piksel")
    ax[1, 1].set_xlim([0, 256])


    # ========================================
    # SIMPAN HISTOGRAM
    # ========================================

    plt.tight_layout()

    nama_histogram = f"hasil/histogram/histogram_{i}.png"

    plt.savefig(
        nama_histogram,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Histogram tersimpan: {nama_histogram}")
    print("-" * 50)


print()
print("SEMUA CITRA BERHASIL DIPROSES!")