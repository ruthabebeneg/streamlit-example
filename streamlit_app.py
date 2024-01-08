import time
import cv2
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from doctr.utils.visualization import visualize_page
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if any(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

# Mode large
st.set_page_config(layout="wide")

# Conception de l'interface
st.title("docTR : Reconnaissance de texte dans les documents")
# Pour une nouvelle ligne
st.write('\n')
# Instructions
st.markdown("*Astuce : cliquez sur le coin supérieur droit d'une image pour l'agrandir !*")
# Définir les colonnes


# Sélection de fichiers
uploaded_files = st.file_uploader("Télécharger des fichiers", type=['pdf', 'png', 'jpeg', 'jpg'], accept_multiple_files=True)


# ...

if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        # Votre logique de traitement pour chaque fichier ici
        if uploaded_file.name.endswith('.pdf'):
            doc = DocumentFile.from_pdf(uploaded_file.read())
        else:
            doc = DocumentFile.from_images(uploaded_file.read())

        for page_idx in range(len(doc)):
            st.write(f"**Page {page_idx + 1}/{len(doc)}**")

            # Sélection du modèle (automatique) - déplacé à l'intérieur de la boucle
            det_arch = "db_resnet50" if doc[page_idx].shape[0] > 1000 else "linknet_resnet18_rotation"
            reco_arch = "crnn_mobilenet_v3_small"

            with st.spinner('Chargement du modèle...'):
                predictor = ocr_predictor(
                    det_arch, reco_arch, pretrained=True,
                    assume_straight_pages=(det_arch != "linknet_resnet18_rotation")
                )

            with st.spinner('Analyse en cours...'):
                # Transmettre l'image au modèle
                processed_batches = predictor.det_predictor.pre_processor([doc[page_idx]])
                out = predictor.det_predictor.model(processed_batches[0], return_model_output=True)
                seg_map = out["out_map"]
                seg_map = tf.squeeze(seg_map[0, ...], axis=[2])
                seg_map = cv2.resize(seg_map.numpy(), (doc[page_idx].shape[1], doc[page_idx].shape[0]),
                                     interpolation=cv2.INTER_LINEAR)
                # Afficher la carte thermique brute
                fig, ax = plt.subplots()
                ax.imshow(seg_map)
                ax.axis('off')

                # Ajouter une barre de progression pour l'analyse OCR
                progress_bar = st.progress(0)
                for i in range(100):
                    # Simuler un processus OCR chronophage
                    time.sleep(0.1)
                    progress_bar.progress(i + 1)

                # Afficher la sortie OCR
                out = predictor([doc[page_idx]])
                fig = visualize_page(out.pages[0].export(), doc[page_idx], interactive=False)

                # Reconstitution de la page sous la page d'entrée
                if det_arch != "linknet_resnet18_rotation":
                    img = out.pages[0].synthesize()

                # Afficher le texte extrait
                text_content = ""
                for block in out.pages[0].blocks:
                    for line in block.lines:
                        for word in line.words:
                            text_content += word.value + " "
                st.write("**Texte extrait :**", text_content)
