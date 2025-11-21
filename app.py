import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Konfiguracja strony
st.set_page_config(
    page_title="Eksplorator Win i Parowania Potraw",
    page_icon="🍷",
    layout="wide"
)

# Tytuł główny
st.title("🍷 Eksplorator Win i Parowania Potraw")
st.markdown("Aplikacja do analizy jakości wina oraz rekomendacji kulinarnych.")

# Funkcja ładowania danych
@st.cache_data
def load_data():
    try:
        # Ładowanie datasetów
        df_pairings = pd.read_csv("wine_food_pairings.csv")
        df_quality = pd.read_csv("winequality-red.csv")
        return df_pairings, df_quality
    except FileNotFoundError:
        st.error("Nie znaleziono plików CSV. Upewnij się, że 'wine_food_pairings.csv' i 'winequality-red.csv' są w katalogu z aplikacją.")
        return None, None

df_pairings, df_quality = load_data()

if df_pairings is not None and df_quality is not None:
    
    # Pasek boczny - Nawigacja
    st.sidebar.header("Nawigacja")
    dataset_choice = st.sidebar.radio(
        "Wybierz moduł analizy:",
        ("Parowanie Wina z Jedzeniem", "Analiza Jakości Wina (Chemia)")
    )

    # --- MODUŁ 1: PAROWANIE WINA Z JEDZENIEM ---
    if dataset_choice == "Parowanie Wina z Jedzeniem":
        st.header("🍽️ Parowanie Wina z Jedzeniem")
        st.write("Znajdź idealne połączenie wina i potrawy w oparciu o typ kuchni i kategorię.")

        # Statystyki ogólne
        col1, col2, col3 = st.columns(3)
        col1.metric("Liczba parowań", df_pairings.shape[0])
        col2.metric("Liczba typów win", df_pairings['wine_type'].nunique())
        col3.metric("Liczba potraw", df_pairings['food_item'].nunique())

        st.markdown("---")

        # Sekcja wyszukiwania
        st.subheader("🔍 Wyszukiwarka Rekomendacji")
        
        search_mode = st.radio("Czego szukasz?", ["Mam wino, szukam potrawy", "Mam potrawę, szukam wina"], horizontal=True)

        if search_mode == "Mam wino, szukam potrawy":
            selected_wine = st.selectbox("Wybierz wino:", sorted(df_pairings['wine_type'].unique()))
            
            # Filtrowanie
            filtered_df = df_pairings[df_pairings['wine_type'] == selected_wine]
            
            # Sortowanie po jakości
            best_pairings = filtered_df.sort_values(by='pairing_quality', ascending=False).head(10)
            
            st.write(f"Najlepsze potrawy do wina **{selected_wine}**:")
            st.dataframe(best_pairings[['food_item', 'food_category', 'cuisine', 'pairing_quality', 'quality_label', 'description']], use_container_width=True)

        else: # Mam potrawę
            selected_food = st.selectbox("Wybierz potrawę:", sorted(df_pairings['food_item'].unique()))
            
            # Filtrowanie
            filtered_df = df_pairings[df_pairings['food_item'] == selected_food]
            
            # Sortowanie po jakości
            best_pairings = filtered_df.sort_values(by='pairing_quality', ascending=False).head(10)
            
            st.write(f"Najlepsze wina do potrawy **{selected_food}**:")
            st.dataframe(best_pairings[['wine_type', 'wine_category', 'pairing_quality', 'quality_label', 'description']], use_container_width=True)

        st.markdown("---")
        
        # Analiza wizualna
        st.subheader("📊 Analiza Trendów")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.write("**Średnia jakość parowania wg Kuchni**")
            cuisine_quality = df_pairings.groupby('cuisine')['pairing_quality'].mean().reset_index().sort_values(by='pairing_quality', ascending=False)
            fig_cuisine = px.bar(cuisine_quality, x='cuisine', y='pairing_quality', color='pairing_quality', color_continuous_scale='Viridis')
            st.plotly_chart(fig_cuisine, use_container_width=True)
            
        with chart_col2:
            st.write("**Rozkład ocen jakości parowania**")
            fig_hist = px.histogram(df_pairings, x='quality_label', category_orders={"quality_label": ["Terrible", "Poor", "Neutral", "Good", "Excellent"]})
            st.plotly_chart(fig_hist, use_container_width=True)


    # --- MODUŁ 2: ANALIZA JAKOŚCI WINA ---
    elif dataset_choice == "Analiza Jakości Wina (Chemia)":
        st.header("🧪 Fizykochemiczna Analiza Jakości Wina")
        st.write("Zbadaj jak właściwości chemiczne wpływają na ocenę jakości wina.")

        if st.checkbox("Pokaż surowe dane"):
            st.dataframe(df_quality.head())

        # Korelacja
        st.subheader("🔥 Macierz Korelacji")
        st.write("Sprawdź, które parametry są ze sobą powiązane.")
        
        # Obliczanie korelacji
        corr = df_quality.corr()
        fig_corr, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig_corr)

        st.markdown("---")

        # Interaktywny wykres punktowy
        st.subheader("📈 Eksploracja Zależności")
        col_x, col_y, col_color = st.columns(3)
        
        with col_x:
            x_axis = st.selectbox("Oś X:", df_quality.columns, index=10) # alcohol default
        with col_y:
            y_axis = st.selectbox("Oś Y:", df_quality.columns, index=1) # volatile acidity default
        with col_color:
            color_by = st.selectbox("Koloruj wg:", ['quality', 'pH', 'alcohol'], index=0)

        fig_scatter = px.scatter(
            df_quality, 
            x=x_axis, 
            y=y_axis, 
            color=color_by, 
            size='total sulfur dioxide', 
            hover_data=df_quality.columns,
            title=f"Relacja: {x_axis} vs {y_axis}"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Analiza wpływu na jakość (Boxplot)
        st.subheader("📦 Wpływ parametru na ocenę jakości (Quality)")
        selected_feature = st.selectbox("Wybierz parametr do analizy:", [col for col in df_quality.columns if col != 'quality'])
        
        fig_box = px.box(df_quality, x='quality', y=selected_feature, color='quality', title=f"Rozkład {selected_feature} dla różnych ocen jakości")
        st.plotly_chart(fig_box, use_container_width=True)

# Stopka
st.markdown("---")
st.caption("Aplikacja stworzona na podstawie dostarczonych danych CSV.")