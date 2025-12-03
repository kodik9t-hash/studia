import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # Nowa biblioteka do interaktywnych wykresów

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------------------------------------
# Konfiguracja strony
# ---------------------------------------------------------
st.set_page_config(
    page_title="Wine Analytics & Food Pairings Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🍷 Wine Analytics & Food Pairings Pro")
st.markdown(
    "Rozbudowana aplikacja do eksploracji jakości win oraz parowania win z jedzeniem. "
    "Zawiera nowe wizualizacje oraz porównanie modeli ML."
)

# ---------------------------------------------------------
# Funkcje wczytywania danych
# ---------------------------------------------------------
@st.cache_data
def load_wine_quality(path: str = "winequality-red.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def load_wine_food_pairings(path: str = "wine_food_pairings.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        return None

# ---------------------------------------------------------
# Wczytanie danych
# ---------------------------------------------------------
wine_quality_df = load_wine_quality()
pairings_df = load_wine_food_pairings()

# ---------------------------------------------------------
# Sidebar – wybór modułu
# ---------------------------------------------------------
st.sidebar.header("⚙️ Ustawienia")
module = st.sidebar.radio(
    "Wybierz moduł:",
    options=["Analiza jakości wina", "Parowanie wina z jedzeniem"]
)

# =========================================================
# 1. ANALIZA JAKOŚCI WINA
# =========================================================
if module == "Analiza jakości wina":
    st.subheader("📊 Analiza jakości czerwonych win")
    
    # Sprawdzenie dostępności pliku
    if wine_quality_df is None:
        st.error(
            "Nie udało się wczytać `winequality-red.csv`.\n\n"
            "Upewnij się, że plik znajduje się w tym samym katalogu co `app.py`."
        )
        st.stop()

    df = wine_quality_df.copy()

    # --- Sekcja: Przegląd Danych ---
    with st.expander("🔎 Podgląd danych surowych"):
        st.dataframe(df.head())
        st.write(df.describe().T)

    # --- Sekcja: Nowe Wizualizacje ---
    st.markdown("---")
    st.markdown("### 🎨 Zaawansowane Wizualizacje (Nowość)")
    
    viz_col1, viz_col2 = st.columns(2)
    
    # 1. Box Plot (Wykres Pudełkowy)
    with viz_col1:
        st.markdown("**1. Rozkład cechy względem jakości (Box Plot)**")
        # Wybór cechy (domyślnie alcohol, jeśli istnieje)
        default_idx = df.columns.get_loc("alcohol") if "alcohol" in df.columns else 0
        feature_box = st.selectbox("Wybierz cechę do analizy:", df.columns.drop('quality'), index=default_idx)
        
        # Używamy Plotly dla interaktywności
        fig_box = px.box(df, x="quality", y=feature_box, color="quality", 
                         title=f"Rozkład: {feature_box} vs Quality",
                         color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_box, use_container_width=True)

    # 2. Bubble Chart (Wykres Bąbelkowy 3D-like)
    with viz_col2:
        st.markdown("**2. Relacja 3 zmiennych (Bubble Chart)**")
        
        # Bezpieczne indeksy domyślne
        cols = list(df.columns.drop('quality'))
        idx_x = cols.index("fixed acidity") if "fixed acidity" in cols else 0
        idx_y = cols.index("pH") if "pH" in cols else min(1, len(cols)-1)
        idx_s = cols.index("alcohol") if "alcohol" in cols else min(2, len(cols)-1)

        x_axis = st.selectbox("Oś X:", cols, index=idx_x)
        y_axis = st.selectbox("Oś Y:", cols, index=idx_y)
        size_axis = st.selectbox("Wielkość bąbelka:", cols, index=idx_s)
        
        fig_bubble = px.scatter(df, x=x_axis, y=y_axis, size=size_axis, color="quality",
                                hover_name="quality", size_max=25,
                                title=f"{x_axis} vs {y_axis} (wielkość = {size_axis})",
                                color_continuous_scale="Viridis")
        st.plotly_chart(fig_bubble, use_container_width=True)

    st.markdown("---")
    
    # --- Sekcja: Stare Wizualizacje (Jako opcja w expanderze) ---
    with st.expander("Klasyczne wizualizacje (Histogram, Heatmapa)"):
        col_old1, col_old2 = st.columns(2)
        with col_old1:
            st.markdown("**Histogram jakości**")
            fig, ax = plt.subplots()
            ax.hist(df["quality"], bins=range(int(df["quality"].min()), int(df["quality"].max()) + 2), edgecolor="black", color="#800020")
            st.pyplot(fig)
        with col_old2:
            st.markdown("**Macierz korelacji**")
            fig_corr, ax_corr = plt.subplots()
            sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm", ax=ax_corr)
            st.pyplot(fig_corr)

    # --- Sekcja: Modelowanie ML ---
    st.markdown("### 🤖 Modele Predykcyjne (ML)")
    
    col_ml1, col_ml2 = st.columns([1, 2])
    
    with col_ml1:
        st.info("Konfiguracja modelu")
        model_type = st.radio("Wybierz algorytm:", ["Random Forest", "Gradient Boosting (Nowy!)"])
        
        test_size = st.slider("Zbiór testowy (%)", 10, 50, 20) / 100.0
        
        # Parametry zależne od modelu
        n_estimators = st.slider("Liczba estymatorów", 50, 500, 200, 50)
        
        learning_rate = 0.1
        max_depth = 3
        if model_type == "Gradient Boosting (Nowy!)":
            learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
            max_depth = st.slider("Max Depth", 1, 10, 3)

    with col_ml2:
        # Przygotowanie danych
        X = df.drop("quality", axis=1)
        y = df["quality"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Wybór i trening modelu
        model = None
        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        else:
            model = GradientBoostingRegressor(
                n_estimators=n_estimators, 
                learning_rate=learning_rate, 
                max_depth=max_depth, 
                random_state=42
            )
            
        with st.spinner(f"Trenowanie modelu {model_type}..."):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
        
        # Wyniki
        st.success(f"Model wytrenowany: **{model_type}**")
        res_c1, res_c2 = st.columns(2)
        res_c1.metric("R² Score (dokładność)", f"{r2:.3f}")
        res_c2.metric("MAE (średni błąd)", f"{mae:.3f}")
        
        # Wykres Rzeczywiste vs Przewidywane
        fig_pred = px.scatter(x=y_test, y=y_pred, labels={'x': 'Rzeczywista jakość', 'y': 'Przewidywana jakość'},
                              title="Wykres: Rzeczywistość vs Predykcja", trendline="ols")
        st.plotly_chart(fig_pred, use_container_width=True)

    # --- Feature Importance ---
    st.markdown("#### Ważność cech dla modelu")
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    fig_imp = px.bar(importances, x='Importance', y='Feature', orientation='h', color='Importance')
    st.plotly_chart(fig_imp, use_container_width=True)
    
    # --- Symulator ---
    st.markdown("#### 🔮 Symulator jakości")
    with st.form("sim_form"):
        cols = st.columns(4)
        user_input = {}
        for i, col_name in enumerate(X.columns):
            with cols[i % 4]:
                user_input[col_name] = st.number_input(col_name, value=float(df[col_name].mean()))
        
        submit = st.form_submit_button("Oblicz prognozę")
        
        if submit:
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)[0]
            st.metric(label="Przewidywana ocena jakości:", value=f"{prediction:.2f}")

# =========================================================
# 2. PAROWANIE WINA Z JEDZENIEM
# =========================================================
elif module == "Parowanie wina z jedzeniem":
    st.subheader("🍽️ Parowanie wina z jedzeniem")
    
    # Sprawdzenie dostępności pliku
    if pairings_df is None:
        st.error(
            "Nie udało się wczytać `wine_food_pairings.csv`.\n\n"
            "Upewnij się, że plik znajduje się w tym samym katalogu co `app.py`."
        )
        st.stop()
    
    dfp = pairings_df.copy()
    
    # --- Sekcja: Filtrowanie ---
    with st.expander("🔍 Filtry wyszukiwania", expanded=True):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            cuisine_sel = st.multiselect("Kuchnia:", options=sorted(dfp["cuisine"].unique()))
        with col_f2:
            food_cat_sel = st.multiselect("Kategoria jedzenia:", options=sorted(dfp["food_category"].unique()))
        with col_f3:
            wine_sel = st.multiselect("Typ wina:", options=sorted(dfp["wine_type"].unique()))
            
        if cuisine_sel: dfp = dfp[dfp["cuisine"].isin(cuisine_sel)]
        if food_cat_sel: dfp = dfp[dfp["food_category"].isin(food_cat_sel)]
        if wine_sel: dfp = dfp[dfp["wine_type"].isin(wine_sel)]

    st.write(f"Liczba pasujących rekordów: {len(dfp)}")
    st.dataframe(dfp.head(10))

    # --- Sekcja: Nowa Wizualizacja (Sunburst) ---
    st.markdown("---")
    st.markdown("### ☀️ Hierarchia Smaków (Sunburst Chart)")
    st.info("Ten wykres pokazuje jak rozkładają się kategorie kuchni, jedzenia i pasujące do nich wina.")
    
    # Przygotowanie danych do Sunburst (musi mieć niezerowe wartości)
    sunburst_data = dfp.groupby(['cuisine', 'food_category', 'wine_type']).size().reset_index(name='count')
    
    if not sunburst_data.empty:
        fig_sun = px.sunburst(
            sunburst_data, 
            path=['cuisine', 'food_category', 'wine_type'], 
            values='count',
            color='cuisine',
            title="Interaktywna mapa parowania: Kuchnia -> Składnik -> Wino"
        )
        st.plotly_chart(fig_sun, use_container_width=True)
    else:
        st.warning("Zbyt mało danych do wygenerowania wykresu po przefiltrowaniu.")

    # --- Sekcja: Statystyki ---
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.markdown("**Top 5 Kuchni w bazie**")
        top_cuisines = dfp['cuisine'].value_counts().head(5)
        st.bar_chart(top_cuisines)
        
    with col_stat2:
        st.markdown("**Rozkład ocen parowania**")
        fig_hist = px.histogram(dfp, x="pairing_quality", nbins=20, title="Histogram jakości dopasowania")
        st.plotly_chart(fig_hist, use_container_width=True)
