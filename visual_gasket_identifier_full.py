
# visual_gasket_identifier_full.py

import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import tempfile
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ---- CONFIGURATION ----
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1DMEdg_WHEriEtwNSok9kiounbbzR2Klq"
CREDENTIALS_PATH = "credentials.json"  # Harus kamu upload

# ---- LOAD GOOGLE SHEET ----
def load_google_sheet(sheet_url):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_PATH, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(sheet_url)
    worksheet = sheet.get_worksheet(0)
    data = worksheet.get_all_records()
    return pd.DataFrame(data)

# ---- IMAGE MATCHING WITH ORB ----
def compare_images(query_img, catalog_images):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(query_img, None)

    best_match = None
    max_matches = 0

    for kode, image_path in catalog_images.items():
        img2 = cv2.imread(image_path, 0)
        if img2 is None:
            continue

        kp2, des2 = orb.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        if des1 is not None and des2 is not None:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            if len(matches) > max_matches:
                max_matches = len(matches)
                best_match = kode

    return best_match

# ---- STREAMLIT APP ----
st.set_page_config(page_title="Visual Gasket Identifier", layout="centered")
st.title("Visual Gasket Identifier (VGI) - Mobile Ready")
st.markdown("Unggah foto gasket atau ambil gambar dari kamera untuk mengenali produk secara otomatis.")

# Load katalog produk dari Google Sheets
try:
    katalog = load_google_sheet(GOOGLE_SHEET_URL)
    st.success("Katalog berhasil dimuat dari Google Sheets")
except Exception as e:
    st.error(f"Gagal memuat katalog: {e}")
    st.stop()

# Buat dict untuk pencocokan gambar berdasarkan KODE
catalog_images = {
    row['KODE']: f"catalog_images/{row['KODE']}.jpg"
    for _, row in katalog.iterrows()
}

# Upload gambar gasket
uploaded_file = st.file_uploader("Upload Foto Gasket", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar Gasket', use_column_width=True)

    # Simpan sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image_path = tmp_file.name
        image.save(image_path)

    img_cv = cv2.imread(image_path, 0)
    result = compare_images(img_cv, catalog_images)

    if result:
        info = katalog[katalog['KODE'] == result].iloc[0]
        st.success(f"🎉 Produk dikenali: {info['NAMA PRODUK']} ({info['KODE']})")
    else:
        st.warning("⚠️ Tidak ditemukan kecocokan dalam katalog.")

st.caption("Dibuat untuk mobile dengan Google Sheet realtime dan pencocokan visual.")
