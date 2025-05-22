import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Zum Laden gespeicherter Modelle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
from PIL import Image
from io import BytesIO


st.set_page_config(
    page_title="My Main Page Title",  # Hier dein benutzerdefinierter Seitentitel
    page_icon=":rocket:",             # Optional, z. B. ein Emoji oder Bild-Link
    layout="centered"                 # Oder "wide", je nach gewünschtem Layout
)



#--------------------------------------------------------------------------------------------------------------------------
# Title
st.title("🏖️ Welcome to the Beach Volleyball Machine Learning Experience!")
st.markdown("## Predict, Analyze, and Win 🏆")

#Pfad festlegen
os.chdir('C:/Users/Katharina/Desktop/Weiterbildung/Bootcamp/Bootcamp/Final_project/streamlit_parts')

# # Konfiguration der Seite
# st.set_page_config(page_title="Meine coole ML App", layout="wide")
# # Streamlit-Titel
# st.title("Machine Learning Model for Beachvolleyball")
# st.subheader('')

# # Erklärung mittels Markdown
# st.markdown("""
# Willkommen zu dieser interaktiven App, die dir verschiedene Machine Learning Modelle präsentiert.  
# Auf **Seite 2** findest du Modelle zur Klassifikation – hier erfährst du, ob ein Team gewinnt oder verliert.  
# Auf **Seite 3** warten Regressionsmodelle, die Metriken wie SpikeFault oder andere numerische Werte vorhersagen.

# Nutze die Navigation in der linken Seitenleiste, um zu den verschiedenen Modellen zu wechseln.  
# Viel Spaß und Erfolg beim Erkunden!
# """)


# # Hinweise / Call-to-Action
# st.info("Verwende die Seitenleiste, um das passende ML Modell auszuwählen und direkt in die Analyse einzusteigen.")
# st.write("Falls du mehr über die Modelle und die Funktionsweise erfahren möchtest, findest du weitere Infos auf den jeweiligen Modellseiten.")
# st.caption("Erstellt von [Dein Name]")

# # Option: Man kann die Seite auch interaktiv gestalten, z.B. mit Buttons, die zu weiteren Infos führen.
# # Ein Button könnte den Nutzer beispielsweise zu einer FAQ oder zu einer Erklärung der Daten leiten.
# if st.button("Mehr über die App erfahren"):
#     st.markdown("""
#     ### Über diese App
#     Diese App wurde entwickelt, um Machine Learning Modelle auf interaktive Weise zu demonstrieren.
#     Hier kannst du direkt in die Vorhersagen einsteigen, Parameter anpassen und live die Auswirkungen sehen.
#     Wähle über die Seitennavigation das passende Modell aus und starte deine Analyse!
#     """)


# Set page config
# st.set_page_config(
#     page_title="Beach Volleyball ML App",
#     page_icon="🏐",
#     layout="centered",
# )

# Introductory text
st.markdown("""
Welcome to this interactive machine learning app focused on **Beach Volleyball**!  
Explore how data and predictive models can enhance performance, strategy, and gameplay insights.

### What can you do here?
- 🧠 Use **Model 1** to analyze team performance and predict match outcomes.  
- 📊 Use **Model 2** to predict the sum of Spike and the sum of spike errors.  
- 🔍 Compare players based on their average statistics.
- 🔍 Compare different Beachvolleyball teams based on their average statistics.

Whether you're a fan, coach, or data scientist — dive in and explore how AI meets beach volleyball!
""")

# Funktion zum Laden von Bildern aus dem Web
def load_image(url):
    resp = requests.get(url)
    return Image.open(BytesIO(resp.content))

# Beispielbilder von Unsplash
img1_url = "https://images.unsplash.com/photo-1723138568659-d35c7680779f?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
# (Beach Volleyball action shot by [Chris Liverani](https://unsplash.com/@chrisliverani))
#img2_url = "https://images.unsplash.com/photo-1587574293340-40d0ca6d9e0e?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=60"
# (Data analysis concept by [Science in HD](https://unsplash.com/@scienceinhd))

# Zeige das einzelne Bild an
st.image(load_image(img1_url),
         caption="Beachvolleyball 🏐",
         use_container_width=True)
# Show example images side by side
# col1, col2 = st.columns(2)
# with col1:
#     st.image(load_image(img1_url), caption="High-level competition 🏐", use_column_width=True)
# with col2:
#     st.image(load_image(img2_url), caption="Data meets sand 📊", use_column_width=True)

# Navigation button
# st.markdown("---")
# if st.button("➡️ Go to the Models"):
#     # Beispiel: Navigiert zur Page “Model 1” im pages-Ordner
#     st.experimental_set_query_params(page="model1")

# Footer
st.markdown("---")
st.caption("Created with ❤️ using Streamlit and Machine Learning.")
