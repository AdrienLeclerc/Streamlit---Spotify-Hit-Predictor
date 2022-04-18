import pandas as pd
import streamlit as st

@st.cache
def load_data():

    ''' loading dataset and creating min / max / mean and artist_mean dataframe for gauge chart '''

    df_artist = pd.read_csv('data/df_artist.csv')
    min_df = pd.DataFrame(df_artist.min()).T
    max_df = pd.DataFrame(df_artist.max()).T
    mean_df = pd.DataFrame(df_artist.mean()).T
    df_artist_mean = df_artist.groupby('artist').mean().reset_index()

    return df_artist, min_df, max_df, mean_df, df_artist_mean