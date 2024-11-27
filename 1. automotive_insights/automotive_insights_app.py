
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



# Angepasster Titel, mittig, größer und unterstrichen
st.markdown(
    """
    <h1 style='text-align: center; font-size: 60px; text-decoration: underline; color: #FFFFFF;'>
        Automotive Insights
    </h1>
    """,
    unsafe_allow_html=True
)


# CSS für dunkelblauen Farbverlauf und karierte Überlagerung mit Waldgrün für die Seitenleiste
page_bg_css = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #283289, #0f1a44), 
                url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='20' height='20'><rect width='10' height='10' fill='%23c0c0c0' opacity='0.2'/><rect x='10' y='10' width='10' height='10' fill='%23c0c0c0' opacity='0.2'/></svg>");
    background-size: cover, 40px 40px;
    background-position: center;
    background-repeat: no-repeat, repeat;
}

[data-testid="stSidebar"] {
    background-color: #26112b; /* Violett für die Sidebar */
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0); /* Header transparent machen */
}

[data-testid="stToolbar"] {
    right: 2rem;
}
</style>
"""

st.markdown(page_bg_css, unsafe_allow_html=True)


st.image("banner1.jpg", use_column_width=True)


# Sidebar
st.sidebar.title("Inhaltsverzeichnis")

# Neuzulassungen
st.sidebar.markdown("## Neuzulassungen")
st.sidebar.markdown("[Neuzulassungen eines spezifischen Landes](#neuzulassungen-eines-spezifischen-landes-anzeigen)")
st.sidebar.markdown("[Neuzulassungen in Europa](#neuzulassungen-in-europa-anzeigen)")

# Durchschnittsverbrauch
st.sidebar.markdown("## Durchschnittsverbrauch")
st.sidebar.markdown("[Durchschnittsverbrauch nach Jahr](#durchschnittsverbrauch-nach-jahr)")

# Kraftstoffarten
st.sidebar.markdown("## Kraftstoffarten")
st.sidebar.markdown("[PKW-Bestand nach Kraftstoffart](#bestand-an-pkw-nach-kraftstoffart-und-jahr)")

# Ladesäulen
st.sidebar.markdown("## Ladesäulen")
st.sidebar.markdown("[Ladesäulen-Entwicklung und Prognose](#ladesäulen-entwicklung-und-prognose-nach-bundesland)")

# Gebrauchtwagen Analyse
st.sidebar.markdown("## Gebrauchtwagen Analyse")
st.sidebar.markdown("[Gebrauchtwagenanalyse nach Marke und Modell](#gebrauchtwagen-analyse-nach-marke-und-modell)")

# Erweiterte Gebrauchtwagen Analyse
st.sidebar.markdown("## Erweiterte Gebrauchtwagen Analyse")
st.sidebar.markdown("[Datenvisualisierung nach Auswahl](#datenvisualisierung-nach-auswahl)")

# Elektro- und Hybridfahrzeuge Analyse
st.sidebar.markdown("## Elektro- und Hybridfahrzeuge")
st.sidebar.markdown("[Top-10 Modelle für Elektro- und Hybridfahrzeuge](#top-10-modelle-fuer-elektro-und-hybridfahrzeuge)")
st.sidebar.markdown("[Detaillierte Informationen zu Elektro- oder Hybridmodellen](#detaillierte-informationen-zu-elektro-oder-hybridmodellen)")

# Machine Learning für Preisvorhersage
st.sidebar.markdown("## Machine Learning für Preisvorhersage")
st.sidebar.markdown("[Preisvorhersage nach Fahrzeugmerkmalen](#preisvorhersage-nach-fahrzeugmerkmalen)")

#Quellenangaben
st.sidebar.markdown("## Quellenverzeichnis")
st.sidebar.markdown("[Quellen](#quellen)", unsafe_allow_html=True)



# Neuzulassungen


# Daten laden
data_neuzulassung = pd.read_csv('neuzulassung.csv', delimiter=';')

# Summiere die Daten aller Länder, um die Gesamtneuzulassungen in Europa zu berechnen
data_neuzulassung['Europe'] = data_neuzulassung.loc[:, 'Belgium':'Kosovo*'] \
    .apply(lambda x: pd.to_numeric(x.astype(str).str.replace('.', ''), errors='coerce')).sum(axis=1)

# Streamlit-Anwendung
st.title('Neuzulassungen nach Antriebsart für Europa und Länder')

st.image("neuwagen.jpg", caption="Neuwagen (AI)", use_column_width=True)

# Aufklappbarer Abschnitt: Auswahl des Landes und der Antriebsarten
st.markdown("<a id='neuzulassungen-eines-spezifischen-landes-anzeigen'></a>", unsafe_allow_html=True)
with st.expander("Neuzulassungen eines spezifischen Landes anzeigen", expanded=False):
    st.subheader("Landesauswahl")

    # Auswahl des Landes
    laender = list(data_neuzulassung.columns[2:-1])  # Ignoriere 'Antrieb', 'Jahr' und 'Europe'
    selected_country = st.selectbox('Wähle das Land aus', laender)

    # Auswahl der Antriebsarten
    antriebe = data_neuzulassung['Antrieb'].unique()
    selected_antriebe = st.multiselect('Wähle die Antriebsarten aus', antriebe, default=antriebe, key="country_antriebsarten")

    # Plot erstellen, wenn mindestens eine Antriebsart ausgewählt wurde
    if selected_antriebe:
        plt.figure(figsize=(12, 8))
        
        for antrieb in selected_antriebe:
            # Filtere die Daten nach Antriebsart und Land
            data_filtered = data_neuzulassung[data_neuzulassung['Antrieb'] == antrieb]
            years = data_filtered['Jahr'].values
            values = pd.to_numeric(data_filtered[selected_country].astype(str).str.replace('.', ''), errors='coerce').fillna(0).values

            # Historische Daten plotten
            sns.lineplot(x=years, y=values, label=f'{antrieb} (Historisch)')
        
        plt.title(f'Neuzulassungen für {selected_country} über die Jahre')
        plt.xlabel('Jahr')
        plt.ylabel('Anzahl der Neuzulassungen')
        plt.legend(title='Antriebsarten')
        plt.grid(True)
        
        # Plot in Streamlit anzeigen
        st.pyplot(plt)
    else:
        st.write("Bitte wähle mindestens eine Antriebsart zur Anzeige aus.")


# Aufklappbarer Abschnitt: Auswahl und Visualisierung für Europa
st.markdown("<a id='neuzulassungen-in-europa-anzeigen'></a>", unsafe_allow_html=True)
with st.expander("Neuzulassungen in Europa anzeigen", expanded=False):
    st.subheader("Europaweite Neuzulassungen")

    # Auswahl der Antriebsarten für Europa mit einem einzigartigen Schlüssel
    selected_antriebe_1 = st.multiselect('Wähle die Antriebsarten für Europa aus', antriebe, default=antriebe, key="europe_antriebsarten")

    # Plot erstellen, wenn mindestens eine Antriebsart ausgewählt wurde
    if selected_antriebe_1:
        plt.figure(figsize=(12, 8))
        
        for antrieb in selected_antriebe_1:
            # Filtere die Daten nach Antriebsart für Europa
            data_filtered = data_neuzulassung[data_neuzulassung['Antrieb'] == antrieb]
            years = data_filtered['Jahr'].values
            values = data_filtered['Europe'].values

            # Historische Daten plotten
            sns.lineplot(x=years, y=values, label=f'{antrieb} (Historisch)')
        
        plt.title('Neuzulassungen in Europa über die Jahre')
        plt.xlabel('Jahr')
        plt.ylabel('Anzahl der Neuzulassungen')
        plt.legend(title='Antriebsarten')
        plt.grid(True)
        
        # Plot in Streamlit anzeigen
        st.pyplot(plt)
    else:
        st.write("Bitte wähle mindestens eine Antriebsart zur Anzeige aus.")





# Daten laden und Komma zu Punkt konvertieren
data_durchschnitt_verbrauch = pd.read_csv('durchschnitt_verbrauch.csv', sep=';')

# Konvertiere alle Verbrauchsspalten zu numerischen Werten, indem Kommas durch Punkte ersetzt werden
for column in data_durchschnitt_verbrauch.columns[1:]:
    data_durchschnitt_verbrauch[column] = pd.to_numeric(
        data_durchschnitt_verbrauch[column].astype(str).str.replace(',', '.'), errors='coerce'
    )

# Streamlit-Anwendung
st.title('Durchschnittsverbrauch nach Jahr')

# Aufklappbarer Abschnitt für die Auswahl und den Plot
st.markdown("<a id='durchschnittsverbrauch-nach-jahr'></a>", unsafe_allow_html=True)
with st.expander("Datenvisualisierung", expanded=False):
    # Auswahl der zu plottenden Spalten
    columns = list(data_durchschnitt_verbrauch.columns[1:])  # Ignoriert die 'Jahr'-Spalte
    
    # Checkbox für das automatische Auswählen aller Spalten
    if 'selected_column' not in st.session_state:
        st.session_state.selected_column = columns  # Startzustand: Alle Spalten ausgewählt

    if st.button("Alle Spalten auswählen / abwählen"):
        # Wenn alle ausgewählt sind, dann Auswahl leeren, ansonsten alle auswählen
        if len(st.session_state.selected_column) == len(columns):
            st.session_state.selected_column = []  # Keine Auswahl
        else:
            st.session_state.selected_column = columns  # Alle auswählen

    # Selectbox für die Einzelspaltenauswahl
    selected_column = st.selectbox('Wähle eine Spalte zum Plotten aus', columns)

    # Checkbox für "Alle plotten"
    plot_all = st.checkbox('Alle Spalten plotten', value=len(st.session_state.selected_column) == len(columns))

    # Plot erstellen
    plt.figure(figsize=(18, 12))
    if plot_all:
        for column in columns:
            sns.lineplot(data=data_durchschnitt_verbrauch, x='Jahr', y=column, label=column)
        # Berechne den Durchschnitt aller Spalten für das erste und letzte Jahr
        start_avg = data_durchschnitt_verbrauch[columns].iloc[0].mean()
        end_avg = data_durchschnitt_verbrauch[columns].iloc[-1].mean()
        reduction_percent = ((start_avg - end_avg) / start_avg) * 100
    else:
        sns.lineplot(data=data_durchschnitt_verbrauch, x='Jahr', y=selected_column, label=selected_column)
        # Berechne die prozentuale Änderung für die ausgewählte Spalte
        start_value = data_durchschnitt_verbrauch[selected_column].iloc[0]
        end_value = data_durchschnitt_verbrauch[selected_column].iloc[-1]
        reduction_percent = ((start_value - end_value) / start_value) * 100

    # Grafiktitel und Achsenbeschriftungen
    plt.title('Verbrauchstrends nach Jahr')
    plt.xlabel('Jahr')
    plt.ylabel('Verbrauch')
    plt.legend(title='Kraftstofftyp')
    plt.grid(True)
    
    # Plot in Streamlit anzeigen
    st.pyplot(plt)
    
    # Prozentrückgang anzeigen
    st.metric(label="Reduzierung des Verbrauchs", value=f"{reduction_percent:.2f}%")



# Kraftstoffarten

st.image("refuel.jpg", caption="Tankstellen", use_column_width=True)


# Daten laden
data_kraftstoff = pd.read_csv('pkw_bestand_kraftstoffart_neu.csv', delimiter=';')

# Streamlit-Anwendung
st.title('Bestand an PKW nach Kraftstoffart und Jahr')

# Aufklappbarer Abschnitt: Auswahl und Visualisierung
st.markdown("<a id='bestand-an-pkw-nach-kraftstoffart-und-jahr'></a>", unsafe_allow_html=True)
with st.expander("Datenvisualisierung", expanded=False):
    # Annahme: Die Datei enthält die Jahre in einer Spalte und die Antriebsarten als Spaltennamen
    # Jahre als x-Achse und Antriebsarten zur Auswahl
    jahre = data_kraftstoff['Jahr']
    antriebe = data_kraftstoff.columns[1:]  # Ignoriere die erste Spalte ('Jahr')

    # Zustand für die Auswahl der Antriebsarten initialisieren
    if 'selected_antriebe' not in st.session_state:
        st.session_state.selected_antriebe = list(antriebe)

    # Button zum Auswählen oder Abwählen aller Antriebsarten
    if st.button("Alle Antriebsarten auswählen / abwählen"):
        if len(st.session_state.selected_antriebe) == len(antriebe):
            st.session_state.selected_antriebe = []  # Auswahl leeren, wenn alle ausgewählt sind
        else:
            st.session_state.selected_antriebe = list(antriebe)  # Alle auswählen

    # Multiselect für die Antriebsarten mit dem session state
    selected_antriebe = st.multiselect(
        'Wähle die Antriebsarten aus',
        antriebe,
        default=st.session_state.selected_antriebe,
        key="antriebsarten_multiselect"
    )

    # Auswahl des Diagrammtyps
    diagramm_typ = st.radio("Diagrammtyp auswählen", ["Liniendiagramm", "Kuchendiagramm", "Balkendiagramm"])

    # Plot erstellen, wenn mindestens eine Antriebsart ausgewählt wurde
    if selected_antriebe:
        plt.figure(figsize=(12, 8))
        
        if diagramm_typ == "Liniendiagramm":
            for antrieb in selected_antriebe:
                # Plotten der Daten für das ausgewählte Antriebsarten
                sns.lineplot(x=jahre, y=data_kraftstoff[antrieb], label=antrieb)
            
            plt.title('PKW-Bestand nach Kraftstoffart über die Jahre')
            plt.xlabel('Jahr')
            plt.ylabel('Bestand')
            plt.legend(title='Antriebsarten')
            plt.grid(True)
            st.pyplot(plt)

        elif diagramm_typ == "Kuchendiagramm":
            # Gesamtsumme für die gewählten Antriebsarten im letzten Jahr
            latest_data = data_kraftstoff[selected_antriebe].iloc[-1]
            plt.pie(latest_data, labels=selected_antriebe, autopct='%1.1f%%', startangle=90)
            plt.title(f"Anteile der Antriebsarten im Jahr {data_kraftstoff['Jahr'].iloc[-1]}")
            st.pyplot(plt)

        elif diagramm_typ == "Balkendiagramm":
            # Daten des letzten Jahres für ein Balkendiagramm
            latest_data = data_kraftstoff[selected_antriebe].iloc[-1]
            plt.bar(selected_antriebe, latest_data)
            plt.title(f"Bestand der Antriebsarten im Jahr {data_kraftstoff['Jahr'].iloc[-1]}")
            plt.xlabel('Antriebsart')
            plt.ylabel('Bestand')
            st.pyplot(plt)

    else:
        st.write("Bitte wähle mindestens eine Antriebsart zur Anzeige aus.")





# Ladesäulen

    
# Daten laden
data_ladesaeulen = pd.read_csv('ladesaeulen.csv')

# Streamlit-Anwendung
st.title('Ladesäulen-Entwicklung und Prognose nach Bundesland')

st.image("Deutschlandkarte1.jpg", caption="Ladeinfrastruktur Deutschland", use_column_width=True)

# Aufklappbarer Abschnitt für die Auswahl und den Plot
st.markdown("<a id='ladesaeulen-entwicklung-und-prognose-nach-bundesland'></a>", unsafe_allow_html=True)
with st.expander("Datenvisualisierung", expanded=False):
    # Auswahl der zu plottenden Spalten
    columns = list(data_ladesaeulen.columns[1:])  # Ignoriere die 'Jahr'-Spalte
    
    # Zustand für die Auswahl der Spalten initialisieren
    if 'selected_column' not in st.session_state:
        st.session_state.selected_column = columns  # Startzustand: Alle Spalten ausgewählt

    # Button zum Auswählen oder Abwählen aller Spalten, mit einem eindeutigen `key`
    if st.button("Alle Spalten auswählen / abwählen", key="toggle_all_columns"):
        # Wenn alle ausgewählt sind, dann Auswahl leeren, ansonsten alle auswählen
        if len(st.session_state.selected_column) == len(columns):
            st.session_state.selected_column = []  # Keine Auswahl
        else:
            st.session_state.selected_column = columns  # Alle auswählen

    # Selectbox für die Einzelspaltenauswahl
    selected_column = st.selectbox('Wähle eine Spalte zum Plotten aus', columns)

    # Checkbox zum Plotten aller Spalten, ebenfalls mit eindeutiger `key`
    plot_all = st.checkbox('Alle Spalten plotten', value=len(st.session_state.selected_column) == len(columns), key="plot_all_columns")

    # Plot erstellen
    plt.figure(figsize=(18, 12))
    
    if plot_all:
        for column in columns:
            sns.lineplot(data=data_ladesaeulen, x='Jahr', y=column, label=column)
        start_avg = data_ladesaeulen[columns].iloc[0].mean()
        end_avg = data_ladesaeulen[columns].iloc[-1].mean()
        reduction_percent = ((end_avg - start_avg) / start_avg) * 100
    else:
        # Nur die ausgewählte Spalte
        sns.lineplot(data=data_ladesaeulen, x='Jahr', y=selected_column, label=selected_column)
        start_value = data_ladesaeulen[selected_column].iloc[0]
        end_value = data_ladesaeulen[selected_column].iloc[-1]
        reduction_percent = ((end_value - start_value) / start_value) * 100

        # Machine Learning für Vorhersage bis 2035
        years = data_ladesaeulen['Jahr'].values.reshape(-1, 1)
        values = data_ladesaeulen[selected_column].values

        # Modell trainieren
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(years, values)

        # Prognose für die Jahre 2024 bis 2035
        future_years = np.arange(2024, 2036).reshape(-1, 1)
        future_predictions = model.predict(future_years)

        # Plot der Prognose
        plt.plot(future_years, future_predictions, '--', label=f'{selected_column} (Prognose)')

        # R²-Score berechnen
        y_pred_train = model.predict(years)
        r2 = r2_score(values, y_pred_train)

        # Berechnung des prozentualen Wachstums bis 2035
        growth_percent = ((future_predictions[-1] - values[0]) / values[0]) * 100

        # Jährliches Wachstum berechnen
        yearly_growth = [(future_predictions[i] - future_predictions[i-1]) / future_predictions[i-1] * 100
                         for i in range(1, len(future_predictions))]

    # Grafiktitel und Achsenbeschriftungen
    plt.title('Ladesäulen-Entwicklung und Prognose nach Jahr')
    plt.xlabel('Jahr')
    plt.ylabel('Anzahl Ladesäulen')
    plt.legend(title='Bundesland')
    plt.grid(True)
    
    # Plot in Streamlit anzeigen
    st.pyplot(plt)
    
    # Anzeige der Wachstumsrate und des R²-Scores
    #st.metric(label="Wachstum bis 2035", value=f"{growth_percent:.2f}%")
    st.metric(label="R²-Score des Modells", value=f"{r2:.2f}")




# Daten laden
data_gebrauchtwagen = pd.read_csv('gebrauchtwagen.csv')

# Streamlit-Anwendung
st.markdown("<a id='gebrauchtwagenanalyse-nach-marke-und-modell'></a>", unsafe_allow_html=True)
st.title('Gebrauchtwagen Analyse nach Marke und Modell')

# Aufklappbarer Abschnitt: Auswahl der Marke und Visualisierung
with st.expander("Datenvisualisierung", expanded=False):
    # Auswahl der Marke über eine Selectbox
    marken = data_gebrauchtwagen['brand'].unique()
    selected_marke = st.selectbox("Wähle eine Marke aus", marken)

    # Filtere die Daten nach der ausgewählten Marke
    data_filtered = data_gebrauchtwagen[data_gebrauchtwagen['brand'] == selected_marke]

    # Berechne die Anzahl der Modelle pro Modelltyp und begrenze auf die Top 15
    model_counts = data_filtered['model'].value_counts().reset_index().head(15)
    model_counts.columns = ['model', 'count']

    # Balkendiagramm mit Seaborn erstellen
    plt.figure(figsize=(12, 8))
    sns.barplot(data=model_counts, x='model', y='count', palette='viridis')
    plt.title(f'Anzahl der Modelle für {selected_marke} (Top 15)')
    plt.xlabel('Modell')
    plt.ylabel('Anzahl der Modelle')
    plt.xticks(rotation=45)
    plt.grid(True)

    # Plot in Streamlit anzeigen
    st.pyplot(plt)




# Konvertiere 'price_in_euro', 'mileage_in_km' und 'power_ps' zu numerischen Werten, entferne nicht konvertierbare Werte
data_gebrauchtwagen['price_in_euro'] = pd.to_numeric(data_gebrauchtwagen['price_in_euro'], errors='coerce')
data_gebrauchtwagen['mileage_in_km'] = pd.to_numeric(data_gebrauchtwagen['mileage_in_km'], errors='coerce')
data_gebrauchtwagen['power_ps'] = pd.to_numeric(data_gebrauchtwagen['power_ps'], errors='coerce')

# Streamlit-Anwendung
st.markdown("<a id='datenvisualisierung-nach-auswahl'></a>", unsafe_allow_html=True)
st.title('Erweiterte Gebrauchtwagen Analyse')

st.image("autoscout24logo.png", use_column_width=True)

# Aufklappbarer Abschnitt für die Diagramme
with st.expander("Datenvisualisierung nach Auswahl", expanded=False):
    
    # Auswahl der Marke und des Modells mit eindeutigen Keys
    selected_brand = st.selectbox("Wähle eine Marke aus", data_gebrauchtwagen['brand'].unique(), key="brand_select")
    filtered_data_by_brand = data_gebrauchtwagen[data_gebrauchtwagen['brand'] == selected_brand]

    selected_model = st.selectbox("Wähle ein Modell aus", filtered_data_by_brand['model'].unique(), key="model_select")
    filtered_data = filtered_data_by_brand[filtered_data_by_brand['model'] == selected_model]

    # Diagramm für Kilometerstände
    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_data['mileage_in_km'], kde=True, color='skyblue')
    plt.title(f'Kilometerstände für {selected_model}')
    plt.xlabel('Kilometerstand')
    plt.ylabel('Anzahl')
    st.pyplot(plt)

    # Diagramm für Farbe
    plt.figure(figsize=(10, 6))
    sns.countplot(data=filtered_data, x='color', palette='Set2')
    plt.title(f'Farben für {selected_model}')
    plt.xlabel('Farbe')
    plt.ylabel('Anzahl')
    st.pyplot(plt)

    # Diagramm für Kraftstoff
    plt.figure(figsize=(10, 6))
    sns.countplot(data=filtered_data, x='fuel_type', palette='husl')
    plt.title(f'Kraftstoffarten für {selected_model}')
    plt.xlabel('Kraftstoff')
    plt.ylabel('Anzahl')
    st.pyplot(plt)

    # Diagramm für Getriebe
    plt.figure(figsize=(10, 6))
    sns.countplot(data=filtered_data, x='transmission_type', palette='Set1')
    plt.title(f'Getriebetypen für {selected_model}')
    plt.xlabel('Getriebe')
    plt.ylabel('Anzahl')
    st.pyplot(plt)

    # Diagramm für Preise
    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_data['price_in_euro'], kde=True, color='orange')
    plt.title(f'Preise für {selected_model}')
    plt.xlabel('Preis (€)')
    plt.ylabel('Anzahl')
    st.pyplot(plt)

    # Diagramm für PS (Leistung)
    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_data['power_ps'], kde=True, color='purple')
    plt.title(f'PS-Werte für {selected_model}')
    plt.xlabel('PS')
    plt.ylabel('Anzahl')
    st.pyplot(plt)




# Konvertiere 'price_in_euro', 'mileage_in_km', 'power_ps' und bereite die Reichweite vor
data_gebrauchtwagen['price_in_euro'] = pd.to_numeric(data_gebrauchtwagen['price_in_euro'], errors='coerce')
data_gebrauchtwagen['mileage_in_km'] = pd.to_numeric(data_gebrauchtwagen['mileage_in_km'], errors='coerce')
data_gebrauchtwagen['power_ps'] = pd.to_numeric(data_gebrauchtwagen['power_ps'], errors='coerce')

# Extrahiere die Reichweite als Zahl aus 'fuel_consumption_g_km'
data_gebrauchtwagen['fuel_consumption_g_km'] = data_gebrauchtwagen['fuel_consumption_g_km'].str.extract(r'(\d+)').astype(float)

# Streamlit-Anwendung
st.title("Analyse für Elektro- und Hybridfahrzeuge")

# Filtere nur Elektro- und Hybridfahrzeuge
data_electric_hybrid = data_gebrauchtwagen[data_gebrauchtwagen['fuel_type'].isin(['Electric', 'Hybrid'])]

# Abschnitt für die Marke und die Top-10 Modelle nach Anzahl
st.markdown("<a id='top-10-modelle-fuer-elektro-und-hybridfahrzeuge'></a>", unsafe_allow_html=True)
with st.expander("Top-10 Modelle für Elektro- und Hybridfahrzeuge", expanded=False):
    # Auswahl der Marke
    marken_electric_hybrid = data_electric_hybrid['brand'].unique()
    selected_brand = st.selectbox("Wähle eine Marke für Elektro- oder Hybridfahrzeuge aus", marken_electric_hybrid)

    # Filtere Daten basierend auf der ausgewählten Marke
    filtered_data = data_electric_hybrid[data_electric_hybrid['brand'] == selected_brand]

    # Top 10 elektrische Modelle nach Anzahl
    st.subheader(f"Top-10 elektrische Modelle für {selected_brand}")
    electric_models = filtered_data[filtered_data['fuel_type'] == 'Electric']['model'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=electric_models.values, y=electric_models.index, palette='Blues_r')
    plt.title(f'Anzahl der elektrischen Modelle für {selected_brand}')
    plt.xlabel('Anzahl')
    plt.ylabel('Elektrische Modelle')
    st.pyplot(plt)

    # Top 10 Hybrid-Modelle nach Anzahl
    st.subheader(f"Top-10 Hybrid-Modelle für {selected_brand}")
    hybrid_models = filtered_data[filtered_data['fuel_type'] == 'Hybrid']['model'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=hybrid_models.values, y=hybrid_models.index, palette='Greens_r')
    plt.title(f'Anzahl der Hybrid-Modelle für {selected_brand}')
    plt.xlabel('Anzahl')
    plt.ylabel('Hybrid-Modelle')
    st.pyplot(plt)

# Abschnitt für Details zu ausgewählten Modellen
st.markdown("<a id='detaillierte-informationen-zu-elektro-oder-hybridmodellen'></a>", unsafe_allow_html=True)
with st.expander("Detaillierte Informationen zu Elektro- oder Hybridmodellen", expanded=False):
    # Filtere Marken, die Elektro- oder Hybridfahrzeuge anbieten
    brands_with_electric_or_hybrid = data_electric_hybrid['brand'].unique()
    selected_brand_details = st.selectbox("Wähle eine Marke für Detailinformationen", brands_with_electric_or_hybrid)

    # Filtere Modelle basierend auf der ausgewählten Marke und Fuel-Typ
    models_with_electric_or_hybrid = data_electric_hybrid[data_electric_hybrid['brand'] == selected_brand_details]['model'].unique()
    selected_model = st.selectbox("Wähle ein elektrisches oder hybrides Modell aus", models_with_electric_or_hybrid)

    # Filtere die Daten für das ausgewählte Modell
    model_data = data_electric_hybrid[(data_electric_hybrid['brand'] == selected_brand_details) & (data_electric_hybrid['model'] == selected_model)]

    # Plot für Preis
    st.subheader(f"Preise für {selected_model}")
    plt.figure(figsize=(10, 6))
    sns.histplot(model_data['price_in_euro'], kde=True, color='orange')
    plt.title(f'Preise für {selected_model}')
    plt.xlabel('Preis (€)')
    plt.ylabel('Anzahl')
    st.pyplot(plt)

    # Plot für Kilometerstand
    st.subheader(f"Kilometerstände für {selected_model}")
    plt.figure(figsize=(10, 6))
    sns.histplot(model_data['mileage_in_km'], kde=True, color='skyblue')
    plt.title(f'Kilometerstände für {selected_model}')
    plt.xlabel('Kilometerstand')
    plt.ylabel('Anzahl')
    st.pyplot(plt)

    # Plot für Farbe
    st.subheader(f"Farben für {selected_model}")
    plt.figure(figsize=(10, 6))
    sns.countplot(data=model_data, x='color', palette='Set2')
    plt.title(f'Farben für {selected_model}')
    plt.xlabel('Farbe')
    plt.ylabel('Anzahl')
    st.pyplot(plt)

    # Plot für PS (Leistung)
    st.subheader(f"PS-Werte für {selected_model}")
    plt.figure(figsize=(10, 6))
    sns.histplot(model_data['power_ps'], kde=True, color='purple')
    plt.title(f'PS-Werte für {selected_model}')
    plt.xlabel('PS')
    plt.ylabel('Anzahl')
    st.pyplot(plt)

    # Plot für Reichweite
    st.subheader(f"Reichweite für {selected_model}")
    plt.figure(figsize=(10, 6))
    sns.histplot(model_data['fuel_consumption_g_km'], kde=True, color='green')
    plt.title(f'Reichweite für {selected_model}')
    plt.xlabel('Reichweite (km)')
    plt.ylabel('Anzahl')
    st.pyplot(plt)


    # Machine Learning


# Konvertiere 'price_in_euro', 'year', 'mileage_in_km' und 'power_ps' zu numerischen Werten
data_gebrauchtwagen['price_in_euro'] = pd.to_numeric(data_gebrauchtwagen['price_in_euro'], errors='coerce')
data_gebrauchtwagen['year'] = pd.to_numeric(data_gebrauchtwagen['year'], errors='coerce')
data_gebrauchtwagen['mileage_in_km'] = pd.to_numeric(data_gebrauchtwagen['mileage_in_km'], errors='coerce')
data_gebrauchtwagen['power_ps'] = pd.to_numeric(data_gebrauchtwagen['power_ps'], errors='coerce')

# Machine Learning Abschnitt
st.header("Machine Learning für Gebrauchtwagen Preisvorhersage")

st.image("ml_forestcar.jpg", caption="Used Cars Random Forest (AI)", use_column_width=True)

st.markdown("<a id='preisvorhersage-nach-fahrzeugmerkmalen'></a>", unsafe_allow_html=True)
with st.expander("Preisvorhersage nach Fahrzeugmerkmalen", expanded=False):
    # Filter für die Marke und das Modell
    selected_brand = st.selectbox("Wähle eine Marke aus", data_gebrauchtwagen['brand'].unique(), key="unique_brand_select")
    filtered_data = data_gebrauchtwagen[data_gebrauchtwagen['brand'] == selected_brand]

    selected_model = st.selectbox("Wähle ein Modell aus", filtered_data['model'].unique(), key="unique_model_select")
    model_data = filtered_data[filtered_data['model'] == selected_model]

    # Wähle die relevanten Features und das Ziel
    features = ['year', 'mileage_in_km', 'power_ps', 'color', 'transmission_type', 'fuel_type']
    target = 'price_in_euro'
    model_data = model_data.dropna(subset=features + [target])  # Entferne Zeilen mit fehlenden Werten

    # Kodierung der kategorischen Variablen
    encoders = {}
    for col in ['color', 'transmission_type', 'fuel_type']:
        le = LabelEncoder()
        model_data[col] = le.fit_transform(model_data[col])
        encoders[col] = le  # Speichere den Encoder mit Klassen für spätere Verwendung

    # Splitte die Daten in Trainings- und Testsets
    X = model_data[features]
    y = model_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Trainiere den Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Vorhersage für jedes Jahr basierend auf den historischen Daten
    year_range = np.arange(model_data['year'].min(), model_data['year'].max() + 1).reshape(-1, 1)
    year_data = pd.DataFrame({'year': year_range.flatten()})
    year_data['mileage_in_km'] = model_data['mileage_in_km'].mean()
    year_data['power_ps'] = model_data['power_ps'].mean()

    # Sicherstellen, dass der häufigste Wert bekannt ist und transformieren, sonst Standardwert verwenden
    for col in ['color', 'transmission_type', 'fuel_type']:
        mode_value = model_data[col].mode()[0]
        if mode_value in encoders[col].classes_:
            year_data[col] = encoders[col].transform([mode_value])[0]
        else:
            year_data[col] = 0  # Verwende einen Standardwert (z.B. 0), falls nicht bekannt

    # Vorhersagen für die Jahresreichweite generieren
    predicted_prices = model.predict(year_data)

    # Plot der Vorhersage und historischen Preise
    st.subheader(f"Preisverlauf und Vorhersage für {selected_model} nach Jahr")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=model_data['year'], y=model_data['price_in_euro'], label='Historische Preise', color='blue', marker='o')
    sns.lineplot(x=year_data['year'], y=predicted_prices, label='Vorhergesagte Preise', color='red', linestyle='--')
    plt.title(f'Preisverlauf für {selected_model} nach Jahr')
    plt.xlabel('Jahr')
    plt.ylabel('Durchschnittlicher Preis (€)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Fehlermetriken anzeigen
    st.subheader("Modell-Fehlermetriken")
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"R²-Score: {r2:.2f}")



    # Quellenverzeichnis in der Hauptansicht

st.header("Quellenverzeichnis")
st.markdown("<a id='quellen'></a>", unsafe_allow_html=True)
with st.expander("Quellen", expanded=False):
    st.markdown(
        """
        - [Deutschlandatlas](https://www.deutschlandatlas.bund.de/DE/Karten/Wie-wir-uns-bewegen/111/_node.html#_t11lwjbxk)
        - [Eurostat Road Transport Data](https://ec.europa.eu/eurostat/databrowser/view/road_eqr_carpda__custom_13451775/default/table?lang=en)
        - [Bundesnetzagentur - E-Mobilität](https://www.bundesnetzagentur.de/DE/Fachthemen/ElektrizitaetundGas/E-Mobilitaet/start.html)
        - [Bundesnetzagentur - Deutschlandkarte](https://www.bundesnetzagentur.de/DE/Fachthemen/ElektrizitaetundGas/E-Mobilitaet/Deutschlandkarte1.jpg?__blob=publicationFile&v=9)
        - [Umweltbundesamt - Durchschnittlicher Kraftstoffverbrauch](https://www.umweltbundesamt.de/bild/durchschnittlicher-kraftstoffverbrauch-von-pkw)
        - [Umweltbundesamt - PKW-Neuzulassungen](https://www.umweltbundesamt.de/bild/entwicklung-der-pkw-neuzulassungen-nach)
        - [Umweltbundesamt - PKW-Bestand nach Kraftstoffart](https://www.umweltbundesamt.de/bild/entwicklung-der-pkw-im-bestand-nach-kraftstoffart)
        - [Kaggle - Germany Used Cars Dataset](https://www.kaggle.com/datasets/wspirat/germany-used-cars-dataset-2023/data)
        - https://www.autoscout24.de/cms-content-assets/1tkbXrmTEPPaTFel6UxtLr-c0eb4849caa00accfa44b32e8da0a2ff-AutoScout24_primary_solid.png
        - https://pixabay.com/de/photos/tanken-zapfsäule-tankstelle-diesel-1629074/

        """
    )