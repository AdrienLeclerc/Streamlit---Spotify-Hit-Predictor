import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from load import *
from plotly.subplots import make_subplots

#loading the data
df_artist, min_df, max_df, mean_df, df_artist_mean = load_data()

@st.cache
def gauge_chart(hit_number, music_number):
    ''' creating gauge chart for a specific column in a dataframe'''
    fig = go.Figure(go.Indicator(
    value = hit_number,
    mode = "gauge+number+delta",
    title = {'text': "Number of hit(s)", 'font': {'size': 18}},
    #delta = {'reference': float(mean_df[df_column]),'increasing': {'color': "#1DB954"}},
    gauge = {'axis': {'range': [0,music_number], 'tickcolor' : 'white'},
             'bar' : {'color' : "#1DB954"},
            'borderwidth': 2,
            'bordercolor': "white"}))

    fig.update_layout(paper_bgcolor = "#0e1117", font = {'color': "white", 'family': "Arial"}, width = 300, height = 300)

    return fig

@st.cache
def gauge_subplot(artist_selected):
    fig = make_subplots(
        rows=5,
        cols=3,
        specs=[[{'type' : 'indicator'}, {'type' : 'indicator'},{'type' : 'indicator'}],
                [{'type' : 'indicator'}, {'type' : 'indicator'},{'type' : 'indicator'}],
                [{'type' : 'indicator'}, {'type' : 'indicator'},{'type' : 'indicator'}],
                [{'type' : 'indicator'}, {'type' : 'indicator'},{'type' : 'indicator'}],
                [{'type' : 'indicator'}, {'type' : 'indicator'},{'type' : 'indicator'}]],
        #horizontal_spacing = 0.01,
        vertical_spacing = 0.02)
    
    fig.add_trace(go.Indicator(
    value = float(artist_selected["danceability"]),
    mode = "gauge+number+delta",
    title = {'text': "Danceability", 'font': {'size': 18}},
    delta = {'reference': float(mean_df["danceability"]),'increasing': {'color': "#1DB954"}},
    gauge = {'axis': {'range': [float(min_df["danceability"]),float(max_df["danceability"])], 'tickcolor' : 'white'},
             'bar' : {'color' : "#1DB954"},
            'borderwidth': 2,
            'bordercolor': "white",
             'steps' : [
                 {'range': [float(min_df["danceability"]), float(mean_df["danceability"])], 'color': "white"}]}), row = 1, col = 1)

    fig.add_trace(go.Indicator(
    value = float(artist_selected["energy"]),
    mode = "gauge+number+delta",
    title = {'text': "Energy", 'font': {'size': 18}},
    delta = {'reference': float(mean_df["energy"]),'increasing': {'color': "#1DB954"}},
    gauge = {'axis': {'range': [float(min_df["energy"]),float(max_df["energy"])], 'tickcolor' : 'white'},
             'bar' : {'color' : "#1DB954"},
            'borderwidth': 2,
            'bordercolor': "white",
             'steps' : [
                 {'range': [float(min_df["energy"]), float(mean_df["energy"])], 'color': "white"}]}), row = 1, col = 2)

    fig.add_trace(go.Indicator(
    value = float(artist_selected["key"]),
    mode = "gauge+number+delta",
    title = {'text': "Key", 'font': {'size': 18}},
    delta = {'reference': float(mean_df["key"]),'increasing': {'color': "#1DB954"}},
    gauge = {'axis': {'range': [float(min_df["key"]),float(max_df["key"])], 'tickcolor' : 'white'},
             'bar' : {'color' : "#1DB954"},
            'borderwidth': 2,
            'bordercolor': "white",
             'steps' : [
                 {'range': [float(min_df["key"]), float(mean_df["key"])], 'color': "white"}]}), row = 1, col = 3)

    fig.add_trace(go.Indicator(
    value = float(artist_selected["loudness"]),
    mode = "gauge+number+delta",
    title = {'text': "Loudness", 'font': {'size': 18}},
    delta = {'reference': float(mean_df["loudness"]),'increasing': {'color': "#1DB954"}},
    gauge = {'axis': {'range': [float(min_df["loudness"]),float(max_df["loudness"])], 'tickcolor' : 'white'},
             'bar' : {'color' : "#1DB954"},
            'borderwidth': 2,
            'bordercolor': "white",
             'steps' : [
                 {'range': [float(min_df["loudness"]), float(mean_df["loudness"])], 'color': "white"}]}), row = 2, col = 1)

    fig.add_trace(go.Indicator(
    value = float(artist_selected["mode"]),
    mode = "gauge+number+delta",
    title = {'text': "Mode", 'font': {'size': 18}},
    delta = {'reference': float(mean_df["mode"]),'increasing': {'color': "#1DB954"}},
    gauge = {'axis': {'range': [float(min_df["mode"]),float(max_df["mode"])], 'tickcolor' : 'white'},
             'bar' : {'color' : "#1DB954"},
            'borderwidth': 2,
            'bordercolor': "white",
             'steps' : [
                 {'range': [float(min_df["mode"]), float(mean_df["mode"])], 'color': "white"}]}), row = 2, col = 2)

    fig.add_trace(go.Indicator(
    value = float(artist_selected["speechiness"]),
    mode = "gauge+number+delta",
    title = {'text': "Speechiness", 'font': {'size': 18}},
    delta = {'reference': float(mean_df["speechiness"]),'increasing': {'color': "#1DB954"}},
    gauge = {'axis': {'range': [float(min_df["speechiness"]),float(max_df["speechiness"])], 'tickcolor' : 'white'},
             'bar' : {'color' : "#1DB954"},
            'borderwidth': 2,
            'bordercolor': "white",
             'steps' : [
                 {'range': [float(min_df["speechiness"]), float(mean_df["speechiness"])], 'color': "white"}]}), row = 2, col = 3)

    fig.add_trace(go.Indicator(
    value = float(artist_selected["acousticness"]),
    mode = "gauge+number+delta",
    title = {'text': "Acoustiness", 'font': {'size': 18}},
    delta = {'reference': float(mean_df["acousticness"]),'increasing': {'color': "#1DB954"}},
    gauge = {'axis': {'range': [float(min_df["acousticness"]),float(max_df["acousticness"])], 'tickcolor' : 'white'},
             'bar' : {'color' : "#1DB954"},
            'borderwidth': 2,
            'bordercolor': "white",
             'steps' : [
                 {'range': [float(min_df["acousticness"]), float(mean_df["acousticness"])], 'color': "white"}]}), row = 3, col = 1)

    fig.add_trace(go.Indicator(
    value = float(artist_selected["instrumentalness"]),
    mode = "gauge+number+delta",
    title = {'text': "Instrumentalness", 'font': {'size': 18}},
    delta = {'reference': float(mean_df["instrumentalness"]),'increasing': {'color': "#1DB954"}},
    gauge = {'axis': {'range': [float(min_df["instrumentalness"]),float(max_df["instrumentalness"])], 'tickcolor' : 'white'},
             'bar' : {'color' : "#1DB954"},
            'borderwidth': 2,
            'bordercolor': "white",
             'steps' : [
                 {'range': [float(min_df["instrumentalness"]), float(mean_df["instrumentalness"])], 'color': "white"}]}), row = 3, col = 2)

    fig.add_trace(go.Indicator(
    value = float(artist_selected["liveness"]),
    mode = "gauge+number+delta",
    title = {'text': "Liveness", 'font': {'size': 18}},
    delta = {'reference': float(mean_df["liveness"]),'increasing': {'color': "#1DB954"}},
    gauge = {'axis': {'range': [float(min_df["liveness"]),float(max_df["liveness"])], 'tickcolor' : 'white'},
             'bar' : {'color' : "#1DB954"},
            'borderwidth': 2,
            'bordercolor': "white",
             'steps' : [
                 {'range': [float(min_df["liveness"]), float(mean_df["liveness"])], 'color': "white"}]}), row = 3, col = 3)

    fig.add_trace(go.Indicator(
    value = float(artist_selected["tempo"]),
    mode = "gauge+number+delta",
    title = {'text': "Tempo", 'font': {'size': 18}},
    delta = {'reference': float(mean_df["tempo"]),'increasing': {'color': "#1DB954"}},
    gauge = {'axis': {'range': [float(min_df["tempo"]),float(max_df["tempo"])], 'tickcolor' : 'white'},
             'bar' : {'color' : "#1DB954"},
            'borderwidth': 2,
            'bordercolor': "white",
             'steps' : [
                 {'range': [float(min_df["tempo"]), float(mean_df["tempo"])], 'color': "white"}]}), row = 4, col = 1)

    fig.add_trace(go.Indicator(
    value = float(artist_selected["duration_ms"]),
    mode = "gauge+number+delta",
    title = {'text': "Duration", 'font': {'size': 18}},
    delta = {'reference': float(mean_df["duration_ms"]),'increasing': {'color': "#1DB954"}},
    gauge = {'axis': {'range': [float(min_df["duration_ms"]),float(max_df["duration_ms"])], 'tickcolor' : 'white'},
             'bar' : {'color' : "#1DB954"},
            'borderwidth': 2,
            'bordercolor': "white",
             'steps' : [
                 {'range': [float(min_df["duration_ms"]), float(mean_df["duration_ms"])], 'color': "white"}]}), row = 4, col = 2)

    fig.add_trace(go.Indicator(
    value = float(artist_selected["time_signature"]),
    mode = "gauge+number+delta",
    title = {'text': "Time Signature", 'font': {'size': 18}},
    delta = {'reference': float(mean_df["time_signature"]),'increasing': {'color': "#1DB954"}},
    gauge = {'axis': {'range': [float(min_df["time_signature"]),float(max_df["time_signature"])], 'tickcolor' : 'white'},
             'bar' : {'color' : "#1DB954"},
            'borderwidth': 2,
            'bordercolor': "white",
             'steps' : [
                 {'range': [float(min_df["time_signature"]), float(mean_df["time_signature"])], 'color': "white"}]}), row = 4, col = 3)

    fig.add_trace(go.Indicator(
    value = float(artist_selected["chorus_hit"]),
    mode = "gauge+number+delta",
    title = {'text': "Chorus Hit", 'font': {'size': 18}},
    delta = {'reference': float(mean_df["chorus_hit"]),'increasing': {'color': "#1DB954"}},
    gauge = {'axis': {'range': [float(min_df["chorus_hit"]),float(max_df["chorus_hit"])], 'tickcolor' : 'white'},
             'bar' : {'color' : "#1DB954"},
            'borderwidth': 2,
            'bordercolor': "white",
             'steps' : [
                 {'range': [float(min_df["chorus_hit"]), float(mean_df["chorus_hit"])], 'color': "white"}]}), row = 5, col = 1)

    fig.add_trace(go.Indicator(
    value = float(artist_selected["sections"]),
    mode = "gauge+number+delta",
    title = {'text': "Sections", 'font': {'size': 18}},
    delta = {'reference': float(mean_df["sections"]),'increasing': {'color': "#1DB954"}},
    gauge = {'axis': {'range': [float(min_df["sections"]),float(max_df["sections"])], 'tickcolor' : 'white'},
             'bar' : {'color' : "#1DB954"},
            'borderwidth': 2,
            'bordercolor': "white",
             'steps' : [
                 {'range': [float(min_df["sections"]), float(mean_df["sections"])], 'color': "white"}]}), row = 5, col = 2)

    fig.add_trace(go.Indicator(
    value = float(artist_selected["valence"]),
    mode = "gauge+number+delta",
    title = {'text': "Valence", 'font': {'size': 18}},
    delta = {'reference': float(mean_df["valence"]),'increasing': {'color': "#1DB954"}},
    gauge = {'axis': {'range': [float(min_df["valence"]),float(max_df["valence"])], 'tickcolor' : 'white'},
             'bar' : {'color' : "#1DB954"},
            'borderwidth': 2,
            'bordercolor': "white",
             'steps' : [
                 {'range': [float(min_df["valence"]), float(mean_df["valence"])], 'color': "white"}]}), row = 5, col = 3)
    
    fig.update_layout(paper_bgcolor = "#0e1117", font = {'color': "white", 'family': "Arial"}, height = 1200, width = 800)

    return fig

@st.cache
def boxplot(df_artist):
    fig = make_subplots(rows=5,cols=3,subplot_titles=("Danceability", "Energy",'Key','Loudness','Mode','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo','Duration','Time Signture','Chorus Hit','Sections'),
        horizontal_spacing = 0.1,
        vertical_spacing = 0.1)

    fig.add_trace(go.Box(y = df_artist['danceability'], x = df_artist['target'], marker_color = "#1DB954"), row = 1, col = 1)
    fig.add_trace(go.Box(y = df_artist['energy'], x = df_artist['target'], marker_color = "#1DB954"), row = 1, col = 2)
    fig.add_trace(go.Box(y = df_artist['key'], x = df_artist['target'], marker_color = "#1DB954"), row = 1, col = 3)

    fig.add_trace(go.Box(y = df_artist['loudness'], x = df_artist['target'], marker_color = "#1DB954"), row = 2, col = 1)
    fig.add_trace(go.Box(y = df_artist['mode'], x = df_artist['target'], marker_color = "#1DB954"), row = 2, col = 2)
    fig.add_trace(go.Box(y = df_artist['speechiness'], x = df_artist['target'], marker_color = "#1DB954"), row = 2, col = 3)

    fig.add_trace(go.Box(y = df_artist['acousticness'], x = df_artist['target'], marker_color = "#1DB954"), row = 3, col = 1)
    fig.add_trace(go.Box(y = df_artist['instrumentalness'], x = df_artist['target'], marker_color = "#1DB954"), row = 3, col = 2)
    fig.add_trace(go.Box(y = df_artist['liveness'], x = df_artist['target'], marker_color = "#1DB954"), row = 3, col = 3)

    fig.add_trace(go.Box(y = df_artist['valence'], x = df_artist['target'], marker_color = "#1DB954"), row = 4, col = 1)
    fig.add_trace(go.Box(y = df_artist['tempo'], x = df_artist['target'], marker_color = "#1DB954"), row = 4, col = 2)
    fig.add_trace(go.Box(y = df_artist['duration_ms'], x = df_artist['target'], marker_color = "#1DB954"), row = 4, col = 3)

    fig.add_trace(go.Box(y = df_artist['time_signature'], x = df_artist['target'], marker_color = "#1DB954"), row = 5, col = 1)
    fig.add_trace(go.Box(y = df_artist['chorus_hit'], x = df_artist['target'], marker_color = "#1DB954"), row = 5, col = 2)
    fig.add_trace(go.Box(y = df_artist['sections'], x = df_artist['target'], marker_color = "#1DB954"), row = 5, col = 3)

    fig.update_layout(
    width = 900,
    height = 1500,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend = False,
    xaxis = dict(showgrid = False),
    yaxis = dict(showgrid = False),
    xaxis1 = dict(showgrid = False),
    yaxis1 = dict(showgrid = False),
    xaxis2 = dict(showgrid = False),
    yaxis2 = dict(showgrid = False),
    xaxis3 = dict(showgrid = False),
    yaxis3 = dict(showgrid = False),
    xaxis4 = dict(showgrid = False),
    yaxis4 = dict(showgrid = False),
    xaxis5 = dict(showgrid = False),
    yaxis5 = dict(showgrid = False),
    yaxis6 = dict(showgrid = False),
    xaxis6 = dict(showgrid = False),
    yaxis7 = dict(showgrid = False),
    xaxis7 = dict(showgrid = False),
    yaxis8 = dict(showgrid = False),
    xaxis8 = dict(showgrid = False),
    yaxis9 = dict(showgrid = False),
    xaxis9 = dict(showgrid = False),
    yaxis10 = dict(showgrid = False),
    xaxis10 = dict(showgrid = False),
    yaxis11 = dict(showgrid = False),
    xaxis11 = dict(showgrid = False),
    yaxis12 = dict(showgrid = False),
    xaxis12 = dict(showgrid = False),
    yaxis13 = dict(showgrid = False),
    xaxis13 = dict(showgrid = False),
    yaxis14 = dict(showgrid = False),
    xaxis14 = dict(showgrid = False),
    yaxis15 = dict(showgrid = False),
    xaxis15 = dict(showgrid = False))

    return fig

@st.cache
def histo(df_artist):
    fig = make_subplots(rows=5,cols=3,subplot_titles=("Danceability", "Energy",'Key','Loudness','Mode','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo','Duration','Time Signture','Chorus Hit','Sections'),
        horizontal_spacing = 0.1,
        vertical_spacing = 0.1)

    fig.add_trace(go.Histogram(x = df_artist['danceability'], marker_color = "#1DB954"), row = 1, col = 1)
    fig.add_trace(go.Histogram(x = df_artist['energy'], marker_color = "#1DB954"), row = 1, col = 2)
    fig.add_trace(go.Histogram(x = df_artist['key'], marker_color = "#1DB954"), row = 1, col = 3)

    fig.add_trace(go.Histogram(x = df_artist['loudness'], marker_color = "#1DB954"), row = 2, col = 1)
    fig.add_trace(go.Histogram(x = df_artist['mode'], marker_color = "#1DB954"), row = 2, col = 2)
    fig.add_trace(go.Histogram(x = df_artist['speechiness'], marker_color = "#1DB954"), row = 2, col = 3)

    fig.add_trace(go.Histogram(x = df_artist['acousticness'], marker_color = "#1DB954"), row = 3, col = 1)
    fig.add_trace(go.Histogram(x = df_artist['instrumentalness'], marker_color = "#1DB954"), row = 3, col = 2)
    fig.add_trace(go.Histogram(x = df_artist['liveness'], marker_color = "#1DB954"), row = 3, col = 3)

    fig.add_trace(go.Histogram(x = df_artist['valence'], marker_color = "#1DB954"), row = 4, col = 1)
    fig.add_trace(go.Histogram(x = df_artist['tempo'], marker_color = "#1DB954"), row = 4, col = 2)
    fig.add_trace(go.Histogram(x = df_artist['duration_ms'], marker_color = "#1DB954"), row = 4, col = 3)

    fig.add_trace(go.Histogram(x = df_artist['time_signature'], marker_color = "#1DB954"), row = 5, col = 1)
    fig.add_trace(go.Histogram(x = df_artist['chorus_hit'], marker_color = "#1DB954"), row = 5, col = 2)
    fig.add_trace(go.Histogram(x = df_artist['sections'], marker_color = "#1DB954"), row = 5, col = 3)

    fig.update_layout(
    width = 900,
    height = 1500,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend = False,
    xaxis = dict(showgrid = False),
    yaxis = dict(showgrid = False),
    xaxis1 = dict(showgrid = False),
    yaxis1 = dict(showgrid = False),
    xaxis2 = dict(showgrid = False),
    yaxis2 = dict(showgrid = False),
    xaxis3 = dict(showgrid = False),
    yaxis3 = dict(showgrid = False),
    xaxis4 = dict(showgrid = False),
    yaxis4 = dict(showgrid = False),
    xaxis5 = dict(showgrid = False),
    yaxis5 = dict(showgrid = False),
    yaxis6 = dict(showgrid = False),
    xaxis6 = dict(showgrid = False),
    yaxis7 = dict(showgrid = False),
    xaxis7 = dict(showgrid = False),
    yaxis8 = dict(showgrid = False),
    xaxis8 = dict(showgrid = False),
    yaxis9 = dict(showgrid = False),
    xaxis9 = dict(showgrid = False),
    yaxis10 = dict(showgrid = False),
    xaxis10 = dict(showgrid = False),
    yaxis11 = dict(showgrid = False),
    xaxis11 = dict(showgrid = False),
    yaxis12 = dict(showgrid = False),
    xaxis12 = dict(showgrid = False),
    yaxis13 = dict(showgrid = False),
    xaxis13 = dict(showgrid = False),
    yaxis14 = dict(showgrid = False),
    xaxis14 = dict(showgrid = False),
    yaxis15 = dict(showgrid = False),
    xaxis15 = dict(showgrid = False))

    return fig

@st.cache
def corr(df_artist):
    y = df_artist['target']
    fields = list(df_artist.columns[:-1])
    correlations = df_artist[fields].corrwith(y)
    correlations.sort_values(inplace=True, ascending = False)
    correlations = pd.DataFrame(correlations)
    correlations['Feature'] = correlations.index
    correlations.columns = ['Correlation','Feature']
    correlations['Correlation'] = round(correlations['Correlation'],2)
    correlations.reset_index(drop=True, inplace = True)
    bar_chart = px.bar(correlations, x = 'Correlation', y = 'Feature', text = "Correlation", title = "Correlations with Target")
    bar_chart.update_layout(
            width = 400,
            height = 400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis = dict(showgrid = False, autorange = "reversed"),
            xaxis = dict(showgrid = False, range = [-0.5, 0.5]),
            font = dict(size = 10, color = 'white'))

    bar_chart.update_traces(marker_color = '#1DB954',
                   textposition = 'outside')

    return bar_chart

@st.cache
def abs_corr(df_artist):
    y = df_artist['target']
    fields = list(df_artist.columns[:-1])
    correlations = df_artist[fields].corrwith(y)
    correlations = pd.DataFrame(correlations)
    correlations['Feature'] = correlations.index
    correlations.columns = ['Correlation','Feature']
    correlations['Correlation'] = round(abs(correlations['Correlation']),2)
    correlations.sort_values(by="Correlation",inplace=True, ascending = False)
    correlations.reset_index(drop=True, inplace = True)
    bar_chart = px.bar(correlations, x = 'Correlation', y = 'Feature', text = "Correlation", title = "Absolute correlations with Target")
    bar_chart.update_layout(
            width = 400,
            height = 400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis = dict(showgrid = False, autorange = "reversed"),
            xaxis = dict(showgrid = False, range = [0, 0.5]),
            font = dict(size = 10, color = 'white'))

    bar_chart.update_traces(marker_color = '#1DB954',
                   textposition = 'outside')

    return bar_chart

@st.cache
def heatmap(df_artist):

    fig = px.imshow(round(df_artist.corr(),2),aspect = "auto", text_auto=True, title = "Correlation Matrix",\
                      labels=dict(x="Feature", y="Feature", color="Correlation"),\
                      color_continuous_scale= ['#F42B2B','#F65252','#0e1117', "#0e1117", "#50BF77", '#34BD64', '#1DB954'])

    return fig
