from load import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


#loading the data
df_artist, min_df, max_df, mean_df, df_artist_mean = load_data()

y = df_artist['target']
X = df_artist.drop(columns = ['target','track','artist'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

MMS_columns = ['danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','sections','valence','tempo','duration_ms','chorus_hit']
OHE_columns = ['key','mode','time_signature','decade']

preprocessor = ColumnTransformer([
    ('Min Max Scaler', MinMaxScaler(), MMS_columns),
    ('One Hot Encoder',OneHotEncoder(handle_unknown='ignore', sparse = False, drop = "first"),OHE_columns)
],
    remainder='drop')

preproc = preprocessor.fit(X_train)
X_train_preproc = preproc.transform(X_train)

RFC = RandomForestClassifier(bootstrap = False, max_depth = 60)
RFC.fit(X_train_preproc, y_train)


