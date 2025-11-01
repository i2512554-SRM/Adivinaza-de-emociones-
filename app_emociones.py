import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator

# Inicializa el modelo de detección de emociones
detector = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Diccionario de emojis según la emoción detectada
emojis = {
    "joy": "😊",
    "anger": "😠",
    "sadness": "😢",
    "fear": "😨",
    "disgust": "🤢",
    "surprise": "😲",
    "neutral": "😐"
}

# Configuración general de la página
st.set_page_config(page_title="Detector de Emociones", page_icon="🧠", layout="centered")

st.title("🧠 Detector de Emociones Multilenguaje")
st.write("Escribe una frase en cualquier idioma y el sistema detectará la emoción usando IA 🤖")

# Entrada del usuario
texto = st.text_area("✍️ Escribe tu frase aquí:")

if st.button("Analizar Emoción"):
    if texto.strip() == "":
        st.warning("⚠️ Por favor, escribe algo antes de analizar.")
    else:
        # Traduce automáticamente al inglés
        traduccion = GoogleTranslator(source="auto", target="en").translate(texto)

        # Analiza emoción
        resultado = detector(traduccion)[0]
        emocion = resultado["label"].lower()
        confianza = resultado["score"]
        emoji = emojis.get(emocion, "🤔")

        # Colores según la emoción
        colores = {
            "joy": "#FFD93D",
            "anger": "#FF6B6B",
            "sadness": "#6A8CAF",
            "fear": "#8D99AE",
            "disgust": "#9DC183",
            "surprise": "#F8C471",
            "neutral": "#D3D3D3"
        }
        color_fondo = colores.get(emocion, "#FFFFFF")

        # Mostrar resultados
        st.markdown(f"<div style='background-color:{color_fondo}; padding:20px; border-radius:10px;'>"
                    f"<h3>📜 Frase original:</h3><p>{texto}</p>"
                    f"<h3>🌐 Traducción:</h3><p>{traduccion}</p>"
                    f"<h3>🧩 Emoción detectada:</h3><p><b>{emocion.capitalize()} {emoji}</b></p>"
                    f"<h3>📊 Confianza:</h3><p>{confianza*100:.2f}%</p>"
                    f"</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Desarrollado por Sebas y Kael 🧠✨")
