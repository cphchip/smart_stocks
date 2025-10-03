# stocks.py
# This script uses the Polygon API to fetch stock data and visualize it using Dash and Plotly.
from polygon import RESTClient
from datetime import date, timedelta, datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, callback, dcc, html
import requests
from stocks_apiKey import news_token, polygon_token
import joblib
import re
import numpy as np
# from math import mean


client = RESTClient(polygon_token)
news_token = news_token
app = Dash()
model = joblib.load("/Users/chiph/Source/side_projects/sentiment_model.pkl")
print("Sentiment model loaded successfully")


# Hold list of ticker symbols
portfolio = [
    'VXUS', 'VOO', 'AAPL', 'VTWO', 'FBTC' 
]
timespan = [
    50, 100, 200
]

##################################### Sentiment Functions #####################################
def load_embeddings(filename):
    """
    Load a DataFrame from the generalized text format used by word2vec, GloVe,
    fastText, and ConceptNet Numberbatch. The main point where they differ is
    whether there is an initial line with the dimensions of the matrix.
    """
    labels = []
    rows = []
    with open(filename, encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            items = line.rstrip().split(' ')
            if len(items) == 2:
                # This is a header row giving the shape of the matrix
                continue
            labels.append(items[0])
            values = np.array([float(x) for x in items[1:]], 'f')
            rows.append(values)
    
    arr = np.vstack(rows)
    return pd.DataFrame(arr, index=labels, dtype='f')


embeddings = embeddings = load_embeddings('/Users/chiph/Source/side_projects/data/wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt')
print("Embeddings loaded successfully")


def vecs_to_sentiment(vecs):
    # predict_log_proba gives the log probability for each class
    predictions = model.predict_log_proba(vecs)

    # To see an overall positive vs. negative classification in one number,
    # we take the log probability of positive sentiment minus the log
    # probability of negative sentiment.
    # this is a logarithm of the max margin for the classifier, 
    # similar to odds ratio (but not exact) log(p_1/p_0) = log(p_1)-log(p_0)
    return predictions[:, 1] - predictions[:, 0]


def words_to_sentiment(words):
    words = [x for x in words if x in embeddings.index]
    # if embeddings.loc[words].dropna() not in embeddings.index:
    #     return pd.DataFrame({'sentiment': [0]}, index=[words])
    # else:
    vecs = embeddings.loc[words].dropna()
    log_odds = vecs_to_sentiment(vecs)
    return pd.DataFrame({'sentiment': log_odds}, index=vecs.index)


TOKEN_RE = re.compile(r"\w.*?\b")


def text_to_sentiment(text): # text = single headline
    # tokenize the input phrase
    tokens = [token.casefold() for token in TOKEN_RE.findall(text)]
    # send each token separately into the embedding, then the classifier
    sentiments = words_to_sentiment(tokens)
    return sentiments['sentiment'].mean() # return the mean for the classifier


###################################### Ticker Data Functions #####################################
def get_news(ticker):
    url = "https://api.benzinga.com/api/v2/news"
    # news_token = news_token
    # ticker = "VOO"
    date_offset = datetime.now().date() - timedelta(days=5)

    querystring = {
        f"token":{news_token},
        "pageSize":"15",
        "displayOutput":"headline",
        "tickers":{ticker},
        "dateFrom":{date_offset}
    }
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers, params=querystring)
    articles = response.json()

    titles = []
    for article in articles:
        if "title" in article:
            titles.append([article["title"], text_to_sentiment(article["title"])])
            # titles.append(article["title"])

    return {ticker: [titles]}


def get_ticker_data(tickers):
    """
    Fetches stock data for the given tickers from the Polygon API.
    Returns a list of DataFrames with the stock data.
    """
    ticker_data_dict = {}
    ticker_news_dict = {}
    DAYS_BACK = 750
    for ticker in tickers:
        ticker_df = pd.DataFrame(
            client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=(date.today() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d"),
                to=date.today().strftime("%Y-%m-%d"),
                adjusted="true",
                sort="asc",
                # limit=750
            )
        )
        ticker_df['timestamp'] = pd.to_datetime(ticker_df['timestamp'], unit='ms')
        ticker_df['date'] = ticker_df['timestamp'].dt.date
        ticker_df.set_index('date', inplace=True)

        # Calcuate Simple Moving Averages (SMA)
        ticker_df['SMA50'] = ticker_df['close'].rolling(window=50).mean()
        ticker_df['SMA200'] = ticker_df['close'].rolling(window=200).mean()

        # Add the DataFrame to the dictionary with the ticker as the key
        ticker_data_dict.update({ticker: ticker_df})
        ticker_news_dict.update(get_news(ticker))

    return ticker_data_dict, ticker_news_dict

ticker_data, ticker_news = get_ticker_data(portfolio)

################################### Dash App Layout and Callbacks #####################################

# Setup the Dash app layout
app.layout = html.Div([
    html.Div(children='Stocks'),
    html.Hr(),
    html.Div([
        html.Div([
            html.H3('Ticker'),
            dcc.RadioItems(options=sorted(portfolio), value='AAPL', id='controls-and-radio'),
        ], style={'margin-right': '40px', 'flex': '1'}),
        html.Div([
            html.H3('Timeframe'),
            dcc.RadioItems(options=timespan, value=200, id='timeframe-radio'),
        ], style={'flex': '1'}),
        html.Div(id='sentiment-label', style={'flex': '1', 'alignSelf': 'center', 'fontWeight': 'bold', 'fontSize': '22px'}),
    ], style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '30px', 'alignItems': 'center'}),
    dcc.Graph(figure={}, id='controls-and-graph', style={'height': '1500px'}),
])
@callback(
    Output(component_id='controls-and-graph', component_property='figure'),
    Output(component_id='sentiment-label', component_property='children'),
    Output(component_id='sentiment-label', component_property='style'),
    Input(component_id='controls-and-radio', component_property='value'),
    Input(component_id='timeframe-radio', component_property='value')
)
def update_graph(selection, timespan):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        row_heights=[0.8, 0.2, 0.4],
        vertical_spacing=.15,
        subplot_titles=(f"{selection} Price", "Volume", "News Headlines"),
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]]
    )

    candle_data = ticker_data[selection].iloc[-timespan:] # just plot the last n selected records
    x_dates = candle_data.index

    # --- OHLC + indicators ---
    fig.add_trace(
        go.Candlestick(
            x=x_dates,
            open=candle_data['open'],
            high=candle_data['high'],
            low=candle_data['low'],
            close=candle_data['close'],
            name=f'{selection} OHLC'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=candle_data['SMA50'],
            mode='lines',
            line=dict(color='blue', width=2),
            name='50-day MA'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=candle_data['SMA200'],
            mode='lines',
            line=dict(color='orange', width=2),
            name='200-day MA'
        ),
        row=1, col=1
    )
    # --- volume ---
    fig.add_trace(
        go.Bar(
            x=x_dates,
            y=candle_data['volume'],
            name='Volume',
            marker_color='rgba(0, 0, 255, 0.5)'
        ),
        row=2, col=1
    )
    # Use date axis and skip weekends (no gaps for non-trading days)
    fig.update_xaxes(
        row=1, col=1,
        rangeslider=dict(visible=True, thickness=0.07),
        rangebreaks=[dict(bounds=["sat", "mon"])]
    )
    fig.update_xaxes(
        row=2, col=1,
        rangebreaks=[dict(bounds=["sat", "mon"])]
    )
    # --- news table ---
    headlines = ticker_news.get(selection)  # get headlines for this ticker
    sentiment_scores = [score[1] for score in headlines[0]] if headlines else []
    if not headlines[0]:
        # headlines = "No news available."
        mean_sentiment = None
        sentiment_text = "No sentiment available"
        sentiment_color = "gray"
    else:
        mean_sentiment = np.mean(sentiment_scores) if sentiment_scores else None
        sentiment_text = f"Mean Sentiment: {mean_sentiment:.2f}"
        sentiment_color = "green" if mean_sentiment > 0 else ("red" if mean_sentiment < 0 else "gray")

    fig.add_trace(
        go.Table(
            header=dict(values=["News Headlines"], fill_color="paleturquoise", align="left"),
            cells=dict(values=headlines, fill_color="lavender", align="left")
        ),
        row=3, col=1
    )
    return fig, sentiment_text, {"color": sentiment_color, "fontWeight": "bold", "fontSize": "22px"}


if __name__ == '__main__':
    app.run(debug=False)