
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import scipy.ndimage as ndi

from skimage import io, measure
from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage.morphology import (
    binary_opening,
    binary_closing,
    remove_small_objects
)
from skimage.measure import regionprops


# ==================================================
# KONFIGURASI STREAMLIT
# ==================================================

st.set_page_config(
    page_title="PCM - ISIC Image Processing",
    page_icon="🩺",
    layout="wide"
)

st.title("Pengolahan Citra Digital - ISIC 2017")
st.write(
    "RGB Channel, Enhancement, Filtering, "
    "Segmentasi, Labeling, dan Ekstraksi Fitur."
)


# ==================================================
# LOKASI DATASET
# ==================================================

file_csv = r"C:\KULIAH\PCM\DATASET ISIC 2017\ISIC-2017_Training_Part3_GroundTruth.csv"
folder_gambar = r"C:\KULIAH\PCM\DATASET ISIC 2017\ISIC-2017_Training_Data"
folder_output = r"C:\KULIAH\PCM\DATASET ISIC 2017\Hasil_Segmentasi"

os.makedirs(folder_output, exist_ok=True)

kelas_pilihan = {
    "melanoma": "Melanoma",
    "seborrheic_keratosis": "Seborrheic Keratosis",
    "nevus": "Nevus (Benign)"
}

target_urutan_kelas = [
    "Melanoma",
    "Seborrheic Keratosis",
    "Nevus (Benign)"
]

kolom_fitur = [
    "sampel_id",
    "image_id",
    "nama_penyakit",
    "label_id",
    "area",
    "centroid_row",
    "centroid_col",
    "major_axis_length",
    "minor_axis_length",
    "eccentricity",
    "orientation"
]


# ==================================================
# MEMBACA DATASET
# ==================================================

@st.cache_data
def baca_csv(path):
    return pd.read_csv(path)


if not os.path.isfile(file_csv):
    st.error("File CSV tidak ditemukan. Periksa file_csv.")
    st.stop()

if not os.path.isdir(folder_gambar):
    st.error("Folder gambar tidak ditemukan. Periksa folder_gambar.")
    st.stop()

data = baca_csv(file_csv)


# ==================================================
# FUNGSI MEMILIH KELAS
# ==================================================

def ambil_data_kelas(key_kelas):

    if key_kelas == "nevus":
        return data[
            (data["melanoma"] == 0.0) &
            (data["seborrheic_keratosis"] == 0.0)
        ]

    return data[data[key_kelas] == 1.0]


def ambil_gambar_kelas(key_kelas):

    data_kelas = ambil_data_kelas(key_kelas)

    for _, baris in data_kelas.iterrows():

        image_id = str(baris["image_id"])

        for ekstensi in [".jpg", ".png"]:

            path_gambar = os.path.join(
                folder_gambar,
                image_id + ekstensi
            )

            if not os.path.isfile(path_gambar):
                continue

            image = cv2.imread(path_gambar)

            if image is not None:
                return cv2.cvtColor(
                    image, cv2.COLOR_BGR2RGB
                )

    return None


# ==================================================
# FUNGSI CONTRAST STRETCHING
# ==================================================

def contrast_stretching(g_original):

    g_original = g_original.astype(np.float32)

    c = np.min(g_original)
    d = np.max(g_original)

    if d - c > 0:
        g_stretched = (
            (g_original - c) * 255.0 / (d - c)
        )
    else:
        g_stretched = g_original.copy()

    return np.clip(
        g_stretched, 0, 255
    ).astype(np.uint8)


# ==================================================
# FUNGSI FILTERING
# ==================================================

@st.cache_data
def proses_filter(g_enhanced):

    tinggi, lebar = g_enhanced.shape

    # Mean Filter - From Scratch
    img_mean = np.zeros(
        (tinggi, lebar), dtype=np.uint8
    )

    for i in range(1, tinggi - 1):
        for j in range(1, lebar - 1):

            jumlah = 0

            for y in range(-1, 2):
                for x in range(-1, 2):
                    jumlah += int(
                        g_enhanced[i + y, j + x]
                    )

            img_mean[i, j] = jumlah // 9

    # Gaussian Filter - From Scratch
    kernel_gauss = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=float) / 16

    img_gauss = np.zeros(
        (tinggi, lebar), dtype=np.uint8
    )

    for i in range(1, tinggi - 1):
        for j in range(1, lebar - 1):

            jumlah = 0.0

            for y in range(-1, 2):
                for x in range(-1, 2):

                    jumlah += (
                        g_enhanced[i + y, j + x]
                        * kernel_gauss[y + 1, x + 1]
                    )

            img_gauss[i, j] = np.clip(
                round(jumlah), 0, 255
            )

    # Non-Linear Filters
    img_median = ndi.median_filter(
        g_enhanced, size=3
    )

    img_min = ndi.minimum_filter(
        g_enhanced, size=3
    )

    img_max = ndi.maximum_filter(
        g_enhanced, size=3
    )

    return (
        img_mean,
        img_gauss,
        img_median,
        img_min,
        img_max
    )


# ==================================================
# FUNGSI SEGMENTASI
# ==================================================

kernel = np.array([
    [False, True, False],
    [True, True, True],
    [False, True, False]
], dtype=bool)


def segmentasi_lesi(img_rgb):

    # 1. Green Channel
    g_original = img_rgb[:, :, 1].astype(np.float32)

    # 2. Contrast Stretching
    c = np.min(g_original)
    d = np.max(g_original)

    if d != c:
        g_enhanced = (
            (g_original - c) * 255.0 / (d - c)
        )
    else:
        g_enhanced = g_original.copy()

    g_enhanced = np.clip(
        g_enhanced, 0, 255
    ).astype(np.uint8)

    # 3. Median Filter
    green_median = ndi.median_filter(
        g_enhanced, size=3
    )

    # 4. Otsu Thresholding
    nilai_otsu = threshold_otsu(green_median)
    otsu_thresh = green_median > nilai_otsu

    # 5. Inversi Mask
    mask_binary = np.logical_not(otsu_thresh)

    # 6. Opening
    opened_mask = binary_opening(
        mask_binary,
        footprint=kernel
    )

    # 7. Menghilangkan Objek Kecil
    cleaned_binary = remove_small_objects(
        opened_mask,
        min_size=30
    )

    # 8. Memilih Komponen Terbesar
    labeled = measure.label(
        cleaned_binary,
        connectivity=2
    )

    if labeled.max() > 0:

        regions = regionprops(labeled)

        region_terbesar = max(
            regions,
            key=lambda region: region.area
        )

        cleaned_binary = (
            labeled == region_terbesar.label
        )

    # 9. Fill Holes
    filled_binary = ndi.binary_fill_holes(
        cleaned_binary
    )

    # 10. Closing
    final_binary = binary_closing(
        filled_binary,
        footprint=kernel
    )

    final_binary = ndi.binary_fill_holes(
        final_binary
    )

    final_mask = (
        final_binary.astype(np.uint8) * 255
    )

    return (
        g_enhanced,
        green_median,
        final_mask,
        nilai_otsu
    )


# ==================================================
# MENU SIDEBAR
# ==================================================

menu = st.sidebar.selectbox(
    "Pilih Tahapan Pengolahan",
    [
        "RGB Channel",
        "Enhancement",
        "Filtering",
        "Segmentasi, Labeling & Ekstraksi Fitur"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("Dataset: ISIC 2017")
st.sidebar.write("Jumlah baris CSV:", len(data))


# ==================================================
# TAHAP 1 - RGB CHANNEL
# ==================================================

if menu == "RGB Channel":

    st.header("Tahap 1 - Analisis RGB Channel")

    fig, axes = plt.subplots(
        3, 4, figsize=(16, 12)
    )

    fig.suptitle(
        "Analisis Channel Warna (R, G, B) dan Histogram",
        fontsize=14,
        fontweight="bold",
        y=0.96
    )

    titles = [
        "1. Image Asli (RGB)",
        "2. Red Channel & Histogram",
        "3. Green Channel & Histogram",
        "4. Blue Channel & Histogram"
    ]

    baris_idx = 0

    for key, nama_kelas in kelas_pilihan.items():

        image_rgb = ambil_gambar_kelas(key)

        if image_rgb is None:
            continue

        channels = [
            (image_rgb[:, :, 0], "#e74c3c"),
            (image_rgb[:, :, 1], "#2ecc71"),
            (image_rgb[:, :, 2], "#3498db")
        ]

        for j in range(4):

            ax = axes[baris_idx, j]

            if j == 0:
                ax.imshow(image_rgb)
                ax.axis("off")

            else:
                ax.axis("off")
                channel, color = channels[j - 1]

                ax_img = ax.inset_axes(
                    [0, 0.35, 1, 0.65]
                )
                ax_img.imshow(channel, cmap="gray")
                ax_img.axis("off")

                ax_hist = ax.inset_axes(
                    [0, 0, 1, 0.30]
                )

                hist, _ = np.histogram(
                    channel.ravel(),
                    bins=256,
                    range=[0, 256]
                )

                ax_hist.bar(
                    range(256), hist,
                    width=1,
                    color=color,
                    alpha=0.8
                )
                ax_hist.set_xlim(0, 256)
                ax_hist.tick_params(labelsize=6)
                ax_hist.grid(
                    True, linestyle="--", alpha=0.2
                )

            if baris_idx == 0:
                ax.set_title(
                    titles[j],
                    fontsize=10,
                    fontweight="bold"
                )

            if j == 0:
                ax.text(
                    -0.25, 0.5, nama_kelas,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontweight="bold"
                )

        baris_idx += 1

    for i in range(baris_idx, 3):
        for j in range(4):
            axes[i, j].axis("off")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ==================================================
# TAHAP 2 - ENHANCEMENT
# ==================================================

elif menu == "Enhancement":

    st.header("Tahap 2 - Enhancement")
    st.write("Contrast Stretching pada Green Channel.")

    fig, axes = plt.subplots(
        3, 3, figsize=(12, 10)
    )

    titles = [
        "1. Image Asli (RGB)",
        "2. Green Channel (Asli)",
        "3. Green Channel (Enhanced)"
    ]

    baris_idx = 0

    for key, nama_kelas in kelas_pilihan.items():

        image_rgb = ambil_gambar_kelas(key)

        if image_rgb is None:
            continue

        g_original = image_rgb[:, :, 1].astype(
            np.float32
        )

        g_enhanced = contrast_stretching(g_original)

        images = [
            (image_rgb, None),
            (g_original.astype(np.uint8), "gray"),
            (g_enhanced, "gray")
        ]

        for j, (img, cmap) in enumerate(images):

            ax = axes[baris_idx, j]

            if cmap:
                ax.imshow(img, cmap=cmap)
            else:
                ax.imshow(img)

            ax.axis("off")

            if baris_idx == 0:
                ax.set_title(
                    titles[j],
                    fontsize=10,
                    fontweight="bold"
                )

            if j == 0:
                ax.text(
                    -0.25, 0.5, nama_kelas,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontweight="bold"
                )

        baris_idx += 1

    for i in range(baris_idx, 3):
        for j in range(3):
            axes[i, j].axis("off")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ==================================================
# TAHAP 3 - FILTERING
# ==================================================

elif menu == "Filtering":

    st.header("Tahap 3 - Filtering")

    titles = [
        "1. Green Enhanced",
        "2. Mean Filter",
        "3. Gaussian Filter",
        "4. Median Filter",
        "5. Min Filter",
        "6. Max Filter"
    ]

    fig, axes = plt.subplots(
        3, 6, figsize=(20, 11)
    )

    fig.suptitle(
        "Perbandingan Filter pada Green Channel Enhanced",
        fontsize=14,
        fontweight="bold"
    )

    baris_idx = 0

    for key, nama_kelas in kelas_pilihan.items():

        image_rgb = ambil_gambar_kelas(key)

        if image_rgb is None:
            continue

        g_original = image_rgb[:, :, 1].astype(
            np.float32
        )

        g_enhanced = contrast_stretching(g_original)

        hasil_filter = proses_filter(g_enhanced)

        images = [
            g_enhanced,
            hasil_filter[0],
            hasil_filter[1],
            hasil_filter[2],
            hasil_filter[3],
            hasil_filter[4]
        ]

        for j, img in enumerate(images):

            ax = axes[baris_idx, j]
            ax.imshow(img, cmap="gray")
            ax.axis("off")

            if baris_idx == 0:
                ax.set_title(
                    titles[j],
                    fontsize=9,
                    fontweight="bold"
                )

            if j == 0:
                ax.text(
                    -0.30, 0.5, nama_kelas,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontweight="bold"
                )

        baris_idx += 1

    for i in range(baris_idx, 3):
        for j in range(6):
            axes[i, j].axis("off")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ==================================================
# TAHAP 4 - SEGMENTASI, LABELING, EKSTRAKSI FITUR
# ==================================================

elif menu == "Segmentasi, Labeling & Ekstraksi Fitur":

    st.header(
        "Tahap 4 - Segmentasi, Labeling & Ekstraksi Fitur"
    )

    st.write(
        "Pipeline: Contrast Stretching → Median Filter "
        "→ Otsu → Inversi → Opening → Remove Small Objects "
        "→ Komponen Terbesar → Fill Holes → Closing."
    )

    # Kernel mengikuti kode asli
    st.caption(
        "Ukuran citra untuk tahap ini: 128 × 128 piksel."
    )

    gambar_pilihan_isic = {}

    # Mengambil satu sampel valid per kelas
    for target_kelas in target_urutan_kelas:

        if target_kelas == "Melanoma":
            data_kelas = data[data["melanoma"] == 1.0]

        elif target_kelas == "Seborrheic Keratosis":
            data_kelas = data[
                data["seborrheic_keratosis"] == 1.0
            ]

        else:
            data_kelas = data[
                (data["melanoma"] == 0.0) &
                (data["seborrheic_keratosis"] == 0.0)
            ]

        for _, row in data_kelas.iterrows():

            image_id = str(row["image_id"])
            img_path = os.path.join(
                folder_gambar, image_id + ".jpg"
            )

            if not os.path.exists(img_path):
                img_path = os.path.join(
                    folder_gambar, image_id + ".png"
                )

            if not os.path.exists(img_path):
                continue

            img_rgb = io.imread(img_path)

            if img_rgb.ndim != 3 or img_rgb.shape[2] < 3:
                continue

            img_rgb = img_rgb[:, :, :3]

            img_rgb = resize(
                img_rgb,
                (128, 128, 3),
                preserve_range=True,
                anti_aliasing=True
            ).astype(np.uint8)

            gambar_pilihan_isic[len(gambar_pilihan_isic)] = {
                "nama_kelas": target_kelas,
                "image_id": image_id,
                "gambar": img_rgb
            }

            break

    st.write(
        f"Sampel valid yang dimuat: "
        f"{len(gambar_pilihan_isic)} dari 3 kelas."
    )

    if not gambar_pilihan_isic:
        st.error("Tidak ada citra yang berhasil dimuat.")

    else:

        semua_fitur = []
        hasil_sampel = []

        # ------------------------------------------
        # SEGMENTASI DAN VISUALISASI
        # ------------------------------------------

        fig, axes = plt.subplots(
            len(gambar_pilihan_isic),
            4,
            figsize=(16, 4 * len(gambar_pilihan_isic)),
            squeeze=False
        )

        titles = [
            "1. Citra Asli (RGB)",
            "2. Green + Contrast Stretching",
            "3. Median Filter",
            "4. Mask Segmentasi"
        ]

        for lbl, data_dict in sorted(
            gambar_pilihan_isic.items()
        ):

            img_rgb = data_dict["gambar"]

            (
                g_enhanced,
                green_median,
                final_mask,
                nilai_otsu
            ) = segmentasi_lesi(img_rgb)

            data_dict["g_enhanced"] = g_enhanced
            data_dict["green_median"] = green_median
            data_dict["final_mask"] = final_mask
            data_dict["nilai_otsu"] = nilai_otsu

            images = [
                (img_rgb, None),
                (g_enhanced, "gray"),
                (green_median, "gray"),
                (final_mask, "gray")
            ]

            for j, (img, cmap) in enumerate(images):

                ax = axes[lbl, j]

                if cmap:
                    ax.imshow(img, cmap=cmap)
                else:
                    ax.imshow(img)

                ax.axis("off")

                if lbl == 0:
                    ax.set_title(
                        titles[j],
                        fontsize=10,
                        fontweight="bold"
                    )

                if j == 0:
                    ax.text(
                        -0.15, 0.5,
                        f"Sampel {lbl}\n"
                        f"{data_dict['nama_kelas'].upper()}",
                        transform=ax.transAxes,
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="bold"
                    )

            # Simpan mask ke folder output
            nama_mask = (
                f"mask_{data_dict['image_id']}.png"
            )

            cv2.imwrite(
                os.path.join(folder_output, nama_mask),
                final_mask
            )

            hasil_sampel.append({
                "sampel_id": lbl,
                "image_id": data_dict["image_id"],
                "nama_penyakit": data_dict["nama_kelas"],
                "nilai_otsu": nilai_otsu,
                "mask": final_mask
            })

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ------------------------------------------
        # LABELING DAN EKSTRAKSI FITUR
        # ------------------------------------------

        st.subheader("Labeling dan Ekstraksi Fitur")

        for lbl, data_dict in sorted(
            gambar_pilihan_isic.items()
        ):

            nama_penyakit = data_dict["nama_kelas"].upper()
            final_mask = data_dict["final_mask"]

            labeled_lesions = measure.label(
                final_mask > 0,
                connectivity=2
            )

            nlabels = int(labeled_lesions.max())

            st.markdown(
                f"**Sampel {lbl}: {nama_penyakit}**"
            )

            col1, col2 = st.columns([1, 2])

            with col1:
                fig_label, ax_label = plt.subplots(
                    figsize=(4, 4)
                )

                ax_label.imshow(
                    labeled_lesions,
                    cmap="nipy_spectral"
                )
                ax_label.set_title(
                    f"Labeling: {nlabels} objek"
                )
                ax_label.axis("off")

                st.pyplot(
                    fig_label,
                    use_container_width=True
                )
                plt.close(fig_label)

            properties = regionprops(labeled_lesions)

            fitur_list = []

            for prop in properties:

                fitur_data = {
                    "sampel_id": lbl,
                    "image_id": data_dict["image_id"],
                    "nama_penyakit": nama_penyakit,
                    "label_id": prop.label,
                    "area": prop.area,
                    "centroid_row": prop.centroid[0],
                    "centroid_col": prop.centroid[1],
                    "major_axis_length": prop.major_axis_length,
                    "minor_axis_length": prop.minor_axis_length,
                    "eccentricity": prop.eccentricity,
                    "orientation": prop.orientation
                }

                fitur_list.append(fitur_data)
                semua_fitur.append(fitur_data)

            df_fitur = pd.DataFrame(
                fitur_list,
                columns=kolom_fitur
            )

            with col2:
                st.write(f"Jumlah objek: {nlabels}")

                if not df_fitur.empty:
                    st.dataframe(
                        df_fitur,
                        use_container_width=True
                    )
                else:
                    st.info(
                        "Tidak ada objek untuk diekstraksi."
                    )

            # CSV untuk masing-masing sampel
            nama_csv = (
                f"fitur_lesi_isic_sampel_{lbl}_"
                f"{nama_penyakit.replace(' ', '_')}.csv"
            )

            df_fitur.to_csv(
                os.path.join(folder_output, nama_csv),
                index=False
            )

        # ------------------------------------------
        # CSV GABUNGAN DAN DOWNLOAD
        # ------------------------------------------

        st.subheader("Ekspor Hasil")

        df_semua_fitur = pd.DataFrame(
            semua_fitur,
            columns=kolom_fitur
        )

        path_csv_gabungan = os.path.join(
            folder_output,
            "fitur_lesi_isic_2017_gabungan.csv"
        )

        df_semua_fitur.to_csv(
            path_csv_gabungan,
            index=False
        )

        st.write(
            f"Total baris fitur yang diekstraksi: "
            f"{len(df_semua_fitur)}"
        )

        st.download_button(
            label="Download CSV Fitur Gabungan",
            data=df_semua_fitur.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="fitur_lesi_isic_2017_gabungan.csv",
            mime="text/csv"
        )

        st.success(
            "Hasil segmentasi, mask, dan CSV fitur telah "
            "disimpan pada folder Hasil_Segmentasi."
        )