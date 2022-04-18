import streamlit as st
import pandas as pd
from load import *
from chart import *
from variables import *
from model import *
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn import set_config; set_config(display='diagram')
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#loading the data
df_artist, min_df, max_df, mean_df, df_artist_mean = load_data()

st.set_page_config(page_title = "Spotify Hit Predictor",
                   page_icon = ":notes:")

with st.sidebar:

    st.image('Images/logo_sidebar.png')

    navigation = option_menu("Navigation", ["Introduction", "Artist Analysis", "Features Analysis", "Modeling", "Hit Predictor"],
                            icons=['house', 'music-note-list', 'bar-chart-line', 'gear','box-arrow-up-right'],
                            menu_icon="app-indicator", default_index=0,
                            styles={
        "container": {"padding": "5!important", "background-color": "#0e1117"},
        "icon": {"color": "#1DB954", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#262730"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

if navigation == "Introduction" : 

    st.image("Images/introduction.png")
    st.markdown("---")
    st.title("Music Features")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image("Images/danceability.png")
        with st.expander('Details') : 
            st.markdown(danceability)
        st.image("Images/mode.png")
        with st.expander('Details') : 
            st.markdown(mode)
        st.image("Images/liveness.png")
        with st.expander('Details') : 
            st.markdown(liveness)
        st.image("Images/time signature.png")
        with st.expander('Details') : 
            st.markdown(time_signature)
    with col2:
        st.image("Images/energy.png")
        with st.expander('Details') : 
            st.markdown(energy)
        st.image("Images/speechiness.png")
        with st.expander('Details') : 
            st.markdown(speechiness)
        st.image("Images/valence.png")
        with st.expander('Details') : 
            st.markdown(valence)
        st.image("Images/chorus hit.png")
        with st.expander('Details') : 
            st.markdown(chorus_hit)
    with col3:
        st.image("Images/key.png")
        with st.expander('Details') : 
            st.markdown(key)
        st.image("Images/acousticness.png")
        with st.expander('Details') : 
            st.markdown(acousticness)
        st.image("Images/tempo.png")
        with st.expander('Details') : 
            st.markdown(tempo)
        st.image("Images/sections.png")
        with st.expander('Details') : 
            st.markdown(sections)
    with col4:
        st.image("Images/loudness.png")
        with st.expander('Details') : 
            st.markdown(loudness)
        st.image("Images/instrumentalness.png")
        with st.expander('Details') : 
            st.markdown(instrumentalness)
        st.image("Images/duration.png")
        with st.expander('Details') : 
            st.markdown(duration_ms)
        st.image("Images/decade.png")
        with st.expander('Details') : 
            st.markdown(decade)



    st.markdown("---")
    st.title("Target : is this song a hit?")
    st.markdown("---")
    st.markdown(target)


if navigation == "Artist Analysis" : 

    st.image("Images/artist analysis.png")
    st.markdown("---")

    artist = st.selectbox(
        "Pick an artist to analyze",
        options = list(df_artist["artist"].unique()))

    artist_selected = df_artist_mean.query("artist == @artist")
    music_artist = df_artist.query("artist == @artist")
    music_number = music_artist.shape[0]
    hit_number = music_artist['target'].sum()
    hit_ratio = int((hit_number / music_number)*100)

    with st.expander("Music features", expanded = True):
        st.markdown(f'These Gauge Charts represent the average features for all the musics available for {artist} (the green gauge) in comparison to the average features for all the dataset (the white gauge).')
        st.plotly_chart(gauge_subplot(artist_selected), use_container_width = True)

    with st.expander("Hit ratio", expanded = True):
        st.markdown(f"On the {music_number} music(s) available on the dataset for {artist}, {hit_number} are considered as a hit which correspond to a hit ratio of {hit_ratio} %.")
        st.plotly_chart(gauge_chart(hit_number, music_number), use_container_width = True)
        
    with st.expander("Show music(s)", expanded = True):
        st.dataframe(music_artist)

if navigation == "Features Analysis" : 

    st.image("Images/features analysis.png")
    st.markdown("---")
    f_analysis = st.selectbox('Choices', ['Features Distributions','Features BoxPlot according to Target', 'Features correlations'])

    if f_analysis == 'Features Distributions':
        st.plotly_chart(histo(df_artist), use_container_width = True)

    if f_analysis == 'Features BoxPlot according to Target':
        st.plotly_chart(boxplot(df_artist), use_container_width = True)
    if f_analysis == 'Features correlations':
        st.plotly_chart(corr(df_artist), use_container_width = True)
        st.plotly_chart(abs_corr(df_artist), use_container_width = True)
        st.plotly_chart(heatmap(df_artist), use_container_width = True)

if navigation == "Modeling" : 

    st.image("Images/modeling.png")
    st.markdown("---")

    with st.expander('Show data', expanded = False):
        st.dataframe(df_artist)
    st.code('''y = df_artist['target']
X = df_artist.drop(columns = ['target','track','artist'])''', language = 'python')

    st.header("Baseline to beat")
    st.markdown('As a baseline we used a Logistic Regression without any feature engineering and hyperparamaters tuning')
    st.code('''log_reg = LogisticRegression(max_iter = 1000)

cross_val_score(log_reg, X, y, cv=5, scoring = 'accuracy')''', language = 'python')
    st.code('''Baseline accuracy : 0.49''', language = 'python')

    st.header("Preprocessing pipeline")

    st.code('''    MMS_columns = ['danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','sections','valence','tempo','duration_ms','chorus_hit']
OHE_columns = ['key','mode','time_signature','decade']''', language = 'python')

    st.code('''    preprocessor = ColumnTransformer([
        ('Min Max Scaler', MinMaxScaler(), MMS_columns),
        ('One Hot Encoder',OneHotEncoder(handle_unknown='ignore', sparse = False, drop = "first"),OHE_columns)
        ],
        remainder='drop')''', language = "python")
    
    st.code('''preproc = preprocessor.fit(X)
X_preproc = preproc.transform(X)''')

    st.header("Cross Val Score on different models")

    st.code('''models = {
    "                   Logistic Regression": LogisticRegression(max_iter = 1000),
    "                   K-Nearest Neighbors": KNeighborsClassifier(),
    "                         Decision Tree": DecisionTreeClassifier(),
    "   Support Vector Machine (RBF Kernel)": SVC(),
    "                         Random Forest": RandomForestClassifier(),
    "                     Gradient Boosting": GradientBoostingClassifier()
}


for name, model in models.items():
    
    accuracy = cross_val_score(model, X_preproc,y,cv=5, scoring = 'accuracy').mean()
    print(name + " cross-val-score mean : " + str(accuracy))''', language = 'python')

    st.code('''- Logistic Regression cross-val-score mean : 0.726
- Gradient Boosting cross-val-score mean : 0.744
- K-Nearest Neighbors cross-val-score mean : 0.657
- Decision Tree cross-val-score mean : 0.676
- Support Vector Machine (RBF Kernel) cross-val-score mean : 0.748
- Random Forest cross-val-score mean : 0.754
- Gradient Boosting cross-val-score mean : 0.744''', language = 'python')

    st.header("RandomForestClassifier hyperparameters tuning")

if navigation == "Hit Predictor" : 

    st.image("Images/hit predictor.png")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.image("Images/danceability.png")
        danceability_input = st.number_input(' ', min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.01, key = "danceability")
        with st.expander('Details') : 
            st.markdown(danceability)

        st.image("Images/mode.png")
        mode_input = st.selectbox(' ', options = [0,1], index = 1)
        with st.expander('Details') : 
            st.markdown(mode)

        st.image("Images/liveness.png")
        liveness_input = st.number_input(' ', min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.01)
        with st.expander('Details') : 
            st.markdown(liveness)

        st.image("Images/time signature.png")
        time_signature_input = st.selectbox(' ', options = [0,1,2,3,4,5], index = 4)
        with st.expander('Details') : 
            st.markdown(time_signature)

    with col2:

        st.image("Images/energy.png")
        energy_input = st.number_input(' ', min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.01, key = "energy")
        with st.expander('Details') : 
            st.markdown(energy)

        st.image("Images/speechiness.png")
        speechiness_input = st.number_input(' ', min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.01, key = "speechiness")
        with st.expander('Details') : 
            st.markdown(speechiness)

        st.image("Images/valence.png")
        valence_input = st.number_input(' ', min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.01, key = "valence")
        with st.expander('Details') : 
            st.markdown(valence)

        st.image("Images/chorus hit.png")
        chorus_input = st.number_input(' ', min_value = 0, max_value = 500, value = 35, step = 1, key = "chorus")
        with st.expander('Details') : 
            st.markdown(chorus_hit)

    with col3:

        st.image("Images/key.png")
        key_input = st.selectbox(' ', options = [0,1,2,3,4,5,6,7,8,9,10,11,12], index = 6)
        with st.expander('Details') : 
            st.markdown(key)

        st.image("Images/acousticness.png")
        accousticness_input = st.number_input(' ', min_value = 0.0, max_value = 1.0, value = 0.38, step = 0.01, key = "accousticness")
        with st.expander('Details') : 
            st.markdown(acousticness)

        st.image("Images/tempo.png")
        tempo_input = st.number_input(' ', min_value = 0, max_value = 300, value = 110, step = 10, key = "tempo")
        with st.expander('Details') : 
            st.markdown(tempo)

        st.image("Images/sections.png")
        section_input = st.number_input(' ', min_value = 0, max_value = 200, value = 10, step = 5, key = "tempo")
        with st.expander('Details') : 
            st.markdown(sections)

    with col4:

        st.image("Images/loudness.png")
        loudness_input = st.number_input(' ', min_value = -60, max_value = 0, value = -10, step = 5, key = "loudness")
        with st.expander('Details') : 
            st.markdown(loudness)

        st.image("Images/instrumentalness.png")
        instrumentalness_input = st.number_input(' ', min_value = 0.0, max_value = 1.0, value = 0.05, step = 0.01, key = "instrumentalness")
        with st.expander('Details') : 
            st.markdown(instrumentalness)

        st.image("Images/duration.png")
        duration_input = st.number_input(' ', min_value = 0, max_value = 4000000, value = 100000, step = 100, key = "duration")
        with st.expander('Details') : 
            st.markdown(duration_ms)

        st.image("Images/decade.png")
        decade_input = st.selectbox(' ', options = [1960,1970,1980,1990,2000,2010], index = 4)
        with st.expander('Details') : 
            st.markdown(decade)
    
    X_new = pd.DataFrame({'danceability' : [float(danceability_input)],
                        'energy' : [float(energy_input)],
                        'key' : [int(key_input)],
                        'loudness' : [float(loudness_input)],
                        'mode' : [int(mode_input)],
                        'speechiness' : [float(speechiness_input)],
                        'acousticness' : [float(accousticness_input)],
                        'instrumentalness' : [float(instrumentalness_input)],
                        'liveness' : [float(liveness_input)],
                        'valence' : [float(valence_input)],
                        'tempo' : [float(tempo_input)],
                        'duration_ms' : [int(duration_input)],
                        'time_signature' : [int(time_signature_input)],
                        'chorus_hit' : [float(chorus_input)],
                        'sections' : [int(section_input)],
                        'decade' : [int(decade_input)]}
                        )
    
    st.markdown('---')

    col1, col2, col3 = st.columns((3,3,3))
        
    with col1 : 
        st.image("Images/thumb.png")

    with col2 : 
        st.image("Images/logo.png")
        predict_button = st.button("---- Hit Prediction Probability ----")

    with col3 : 
        st.image("Images/thumb2.png")

    if predict_button :

        X_new_preproc = preproc.transform(X_new)

        hit_prediction_proba = round(RFC.predict_proba(X_new_preproc)[0][1]*100,0)

        st.markdown(f"<h2 style='text-align: center;'>Probability of being a hit : {hit_prediction_proba} %</h2>", unsafe_allow_html=True)