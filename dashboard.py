from __future__ import division
from math import pi
from bokeh.palettes import Category20c
from bokeh.transform import cumsum
import calendar
import datetime as DT
from datetime import datetime, date
import numpy as np
import pandas as pd
import holoviews as hv
from bokeh.plotting import show, figure, curdoc, output_file
from bokeh.models.tools import PanTool, SaveTool
from bokeh.io import output_file, show, curdoc
from bokeh.layouts import layout, widgetbox, column, row
from bokeh.models import ColumnDataSource, HoverTool, BoxZoomTool, ResetTool, PanTool, WheelZoomTool, SaveTool, LassoSelectTool
from bokeh.models import CustomJS, ColumnDataSource, Slider, DateRangeSlider, DatetimeTickFormatter
from bokeh.models.widgets import Slider, Select, TextInput, Div, DataTable, DateFormatter, TableColumn, Panel, Tabs, Toggle
from bokeh.io import output_file, show
from bokeh.models.widgets import RadioButtonGroup, Button
from bokeh.models.widgets.inputs import DatePicker

import emoji
import pandas as pd
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

from bokeh.io import show
from bokeh.layouts import widgetbox
from bokeh.models.widgets import TextInput

from bokeh.io import output_file, show
from bokeh.layouts import widgetbox
from bokeh.models.widgets import Button

from bokeh.models.widgets import Paragraph

import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup as bs
import time
import pandas as pd
import datetime
from  multiprocessing import Process
import threading

from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Range1d, LabelSet, Label
from bokeh.palettes import Category20, Paired, Spectral

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from wordcloud.wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.corpus import stopwords

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
#nltk.download('vader_lexicon')
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Range1d, LabelSet, Label, HoverTool
from bokeh.palettes import Category20, Paired, Spectral



import numpy as np
from PIL import Image
from bokeh.plotting import figure, show, output_file
from pprint import pprint


from sklearn.feature_extraction.text import CountVectorizer
import lda
import numpy as np
topic_num=12

def init_driver():
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome('/home/dsawlani/chromedriver',chrome_options=chrome_options)
    # set a default wait time for the browser [5 seconds here]:
    driver.wait = WebDriverWait(driver, 5)
    return driver


def close_driver(driver):
 
    driver.close()
 
    return

def login_twitter(driver, username, password):
 
    # open the web page in the browser:
    driver.get("https://twitter.com/login")
 
    # find the boxes for username and password
    username_field = driver.find_element_by_class_name("js-username-field")
    password_field = driver.find_element_by_class_name("js-password-field")
 
    # enter your username:
    username_field.send_keys(username)
    driver.implicitly_wait(1)
 
    # enter your password:
    password_field.send_keys(password)
    driver.implicitly_wait(1)
 
    # click the "Log In" button:
    driver.find_element_by_class_name("EdgeButtom--medium").click()
 
    return


class wait_for_more_than_n_elements_to_be_present(object):
    def __init__(self, locator, count):
        self.locator = locator
        self.count = count
 
    def __call__(self, driver):
        try:
            elements = EC._find_elements(driver, self.locator)
            return len(elements) > self.count
        except StaleElementReferenceException:
            return False
        
        
def search_twitter(driver, query):
 
    # wait until the search box has loaded:
    box = driver.wait.until(EC.presence_of_element_located((By.NAME, "q")))
 
    # find the search box in the html:
    driver.find_element_by_name("q").clear()
 
    # enter your search string in the search box:
    box.send_keys(query)
 
    # submit the query (like hitting return):
    box.submit()
 
    # initial wait for the search results to load
    wait = WebDriverWait(driver, 5)
 
    try:
        # wait until the first search result is found. Search results will be tweets, which are html list items and have the class='data-item-id':
        wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "li[data-item-id]")))
 
        # scroll down to the last tweet until there are no more tweets:
        while True:
 
            # extract all the tweets:
            tweets = driver.find_elements_by_css_selector("li[data-item-id]")
 
            # find number of visible tweets:
            number_of_tweets = len(tweets)
            #print(number_of_tweets)
 
            
            # keep scrolling:
            #driver.execute_script("arguments[0].scrollIntoView();", tweets[-1])
            #print(tweets)"arguments[0].scrollIntoView();"//*[@class='session[username_or_email]']"
            wait = WebDriverWait(driver, 15)
            #elm = wait.until(EC.visibility_of_element_located((By.XPATH, "//*[@class='Icon Icon--large Icon--logo']")))
            #driver.execute_script("arguments[0].scrollIntoView();", elm)
            #driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            #element = driver.find_element_by_class_name("Icon Icon--large Icon--logo")
            #driver.findElement(By.id("home_search_input")).sendKeys("demo");
            #driver.execute_script("arguments[0].scrollIntoView();", element)
            
            SCROLL_PAUSE_TIME = 2

            # Get scroll height
            last_height = driver.execute_script("return document.body.scrollHeight")

            while True:
                # Scroll down to bottom
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

                # Wait to load page
                time.sleep(SCROLL_PAUSE_TIME)

                # Calculate new scroll height and compare with last scroll height
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
            
            
            try:
                # wait for more tweets to be visible:
                wait.until(wait_for_more_than_n_elements_to_be_present(
                    (By.CSS_SELECTOR, "li[data-item-id]"), number_of_tweets))
 
            except TimeoutException:
                # if no more are visible the "wait.until" call will timeout. Catch the exception and exit the while loop:
                break
           
        # extract the html for the whole lot:
        page_source = driver.page_source
 
    except TimeoutException:
 
        # if there are no search results then the "wait.until" call in the first "try" statement will never happen and it will time out. So we catch that exception and return no html.
        page_source=None
 
    return page_source


def extract_tweets(page_source):
    start = time.clock()
    soup = bs(page_source, 'lxml')
 
    tweets = []
    for li in soup.find_all("li", class_='js-stream-item'):
 
        # If our li doesn't have a tweet-id, we skip it as it's not going to be a tweet.
        if 'data-item-id' not in li.attrs:
            continue
 
        else:
            tweet = {
                'emoji':None,
                'tweet_id': li['data-item-id'],
                'text': None,
                #'user_id': None,
                #'user_screen_name': None,
                #'user_name': None,
                'created_at': None,
                #'retweets': 0,
                #'likes': 0,
                #'replies': 0
            }
 
            
            #Emojis
            escratch = []
            emoji_p = li.find_all("img", class_="Emoji--forText")
            for e in emoji_p:
                escratch.append(e.get("alt"))
            tweet['emoji'] = "".join(escratch)
            
            # Tweet Text
            text_p = li.find("p", class_="tweet-text")
            if text_p is not None:
                tweet['text'] = text_p.get_text()
            '''
            # Tweet User ID, User Screen Name, User Name
            user_details_div = li.find("div", class_="tweet")
            if user_details_div is not None:
                tweet['user_id'] = user_details_div['data-user-id']
                tweet['user_screen_name'] = user_details_div['data-screen-name']
                tweet['user_name'] = user_details_div['data-name']
             '''
            # Tweet date
            date_span = li.find("span", class_="_timestamp")
            if date_span is not None:
                tweet['created_at'] = datetime.datetime.fromtimestamp(int(date_span['data-time-ms']) / 1e3).date()
            '''
            # Tweet Retweets
            retweet_span = li.select("span.ProfileTweet-action--retweet > span.ProfileTweet-actionCount")
            if retweet_span is not None and len(retweet_span) > 0:
                tweet['retweets'] = int(retweet_span[0]['data-tweet-stat-count'])
 
            # Tweet Likes
            like_span = li.select("span.ProfileTweet-action--favorite > span.ProfileTweet-actionCount")
            if like_span is not None and len(like_span) > 0:
                tweet['likes'] = int(like_span[0]['data-tweet-stat-count'])
 
            # Tweet Replies
            reply_span = li.select("span.ProfileTweet-action--reply > span.ProfileTweet-actionCount")
            if reply_span is not None and len(reply_span) > 0:
                tweet['replies'] = int(reply_span[0]['data-tweet-stat-count'])
             '''
            tweets.append(tweet)
    stop = time.clock()
    return tweets

noptions = Select(title="Number of Tweets", options=['100', '500', '1000', '5000'], value='100', width = 200)
radio_button_group = RadioButtonGroup(labels=["Hashtag", "Search Term"], active=0, width = 200)
text_input = TextInput(value="", width = 200, title = "Search Term")
button = Button(label="Submit", width = 200)
buttoncomb = row(button, Div(text = ""))

t = [] 
def logic(query):
    #t = []
    
    driver = init_driver()
    username = 'dhavalsawlani1995@gmail.com'
    password = 'iloveugod'
    login_twitter(driver, username, password)
    #query = "#trump since:2018-10-01"
    page_source = search_twitter(driver, query)
    tweets = extract_tweets(page_source)
    close_driver(driver)
    
    
    t.append(tweets)

def get_tweets():
    start = time.clock()
    N = 6
    dates = []
    startdate = (datetime.datetime.now() - datetime.timedelta(1))
    for i in range(N):
        dates.append((startdate - datetime.timedelta(i)).date())
    query = []
    searchterm = text_input.value
    for i in range(N):
        try:
            query.append(searchterm + " since:" + str(dates[i+1]) + " until:" + str(dates[i]))
        except:
            pass
    #print('logs_' + searchterm + '_' + str(datetime.datetime.now()).split('.')[0]+ '.txt')
    now = datetime.datetime.now()
    log_file = open(searchterm + ' ' + str(now.month) + '-'+ str(now.day) +'-'+ str(now.year)+' '+
                    str(now.hour)+'-'+str(now.minute)+'-'+str(now.second)+ '.txt', 'w')
    # Number of browsers to spawn
    thread_list = list()
    tt = []
    # Start test
    NN = len(query)
    for i in range(NN):
        th = threading.Thread(name='Test {}'.format(i), target=logic, args = [query[i]])
        th.start()
        time.sleep(2)
        log_file.write(th.name + ' started on: '+ str(datetime.datetime.now()).split('.')[0]+'\n\n')
        log_file.write(query[i]+'\n\n')
        thread_list.append(th)
    
    # Wait for all thre<ads to complete
    for thread in thread_list:
        thread.join()
    end = time.clock()
    log_file.write('Test completed on: '+ str(datetime.datetime.now()).split('.')[0]+'\n\n')
    log_file.write("Tweets extracted: " + str(sum([len(t[i]) for i in range(len(t))])) + "\n")
    log_file.write('Time taken: '+ str((end-start)/60))
    log_file.close()
    return t

source = ColumnDataSource(data=dict(Hashtag = [], Date = [], Tweet = []))
columns = [
TableColumn(field="Hashtag", title="Hashtag"),
TableColumn(field="Date", title="Date", formatter = DateFormatter()),
TableColumn(field="Tweet", title="Tweet"),
]
data_table = DataTable(source=source, columns=columns, width=550, height=400)
column2 = column([])

source_emoji = ColumnDataSource(data=dict(emoji = [], Count = [], Rank = []))
columns_emoji = [
TableColumn(field="emoji", title="emoji"),
TableColumn(field="Count", title="Count"),
TableColumn(field="Rank", title="Rank")
]
data_table_emoji = DataTable(source=source_emoji, columns=columns_emoji, width=550, height=400)
p = figure(title="Popular Emojis", toolbar_location=None, tools="", plot_height = 500, plot_width = 750)
hover = HoverTool(
        tooltips=[('Rank', '@Rank'), ('Emoji', '@emoji'), ('Count', '@Count')],
        mode = 'vline'
        )

p1 = figure(title="Popular Words", toolbar_location=None, tools="", plot_height = 500, plot_width = 750)
hover1 = HoverTool(
        tooltips=[('Rank', '@Rank'), ('Word', '@Word'), ('Count', '@Count')]
        )

fig = figure(title="Most Frequent Singular Words" ,x_range=(0,500), y_range=(0,750),
             plot_height = 500, plot_width = 750, tools="")
fig.toolbar.logo = None
fig.axis.axis_label=None
fig.axis.visible=False
fig.grid.grid_line_color = None
fig.xaxis.minor_tick_line_color = None
fig.xaxis.major_tick_line_color = None
fig.yaxis.minor_tick_line_color = None
fig.yaxis.major_tick_line_color = None

sent = figure(title="Sentiment Polarity", toolbar_location=None, tools="", plot_height = 500, plot_width = 750)

plot_events = figure(plot_width=750, plot_height=500, tools = "", x_range=(-0.5, 1.0), toolbar_location = None,
                                  title = "Emotion Radar")
plot_events.toolbar.logo = None
plot_events.axis.axis_label=None
plot_events.axis.visible=False
plot_events.grid.grid_line_color = None
plot_events.xaxis.minor_tick_line_color = None
plot_events.xaxis.major_tick_line_color = None
hover2 = HoverTool(
        tooltips=[('Emotion', '@Emotion'), ('Percentage', '@Percentage')]
        )
def text_clean(message):
    nopunc = [i for i in message if i not in string.punctuation]
    nn = "".join(nopunc)
    nn = nn.lower().split()
    nostop = [words for words in nn if words not in stopwords.words('english')]
    return(nostop)


def on_buttonpress():
    if text_input.value != "":
        tweets = get_tweets()
        allemojis = []
        ef = []
        for i in range(len(t)):
            rr = len(t[i])
            for j in range(rr):
                allemojis = "".join(t[i][j]['emoji'])
                emoji_list = emoji.emoji_lis(allemojis)
                if emoji_list != []:
                    ef.append(emoji_list)
                em = []
                for k in range(len(ef)):
                    for l in range(len(ef[k])):
                        em.append(ef[k][l]['emoji'])
                emoji_series = pd.Series(em)
                emojis = pd.DataFrame(emoji_series.value_counts()).reset_index().rename(columns = {'index':'emoji', 0: 'Count'})
                emojis['Rank'] = pd.Series(range(1, len(emojis)))
        emojis = emojis.head(10)
        emojis['Rank'] = emojis['Rank'].apply(lambda x: int(x))
        source_emoji.data = dict(
            emoji = emojis['emoji'],
            Count = emojis['Count'],
            Rank  = emojis['Rank'],
            color = Paired[10]
        )
        labels = LabelSet(x="Rank", y="Count", text="emoji", level='glyph', render_mode='canvas', source = source_emoji,
                          x_offset = -16, y_offset = -14, text_font_size="23pt")
        p.vbar(x="Rank", top="Count", width = 0.95, source = source_emoji, color = "color")
        p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
        p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
        p.y_range.start = 0
        p.x_range.start = 0
        p.xaxis[0].ticker.desired_num_ticks = 10
        p.add_layout(labels)
        p.xaxis.minor_tick_line_color = None
        p.xgrid.visible = False
        p.ygrid.visible = False
        p.xaxis.major_tick_line_color = None
        p.add_tools(hover)
        
        ####wordcloud
        alltweets = []
        for j in range(len(t)):
            for i in range(len(t[j])):
                alltweets.append(t[j][i]['text'])
        combined_tweets = "".join(alltweets)
        cleaned_tweets = text_clean(combined_tweets)
        ss = pd.DataFrame(pd.Series(cleaned_tweets).value_counts()).reset_index().rename(columns={'index':'Word', 0:'Count'})
        ss['Rank'] = pd.Series(range(1, len(ss)))
        ss = ss.head(10)
        ss['Rank'] = ss['Rank'].apply(lambda x: int(x))
        
        
        data_text = dict(Word = list(ss.Word) , Count = list(ss.Count), Rank = list(ss.Rank), color = Category20[10])
        #labels_text = LabelSet(x="Rank", y="Count", text="Word", level='glyph', render_mode='css', source = ColumnDataSource(data_text),
                               #x_offset = -7, text_font_size="10pt", y_offset = 7, angle = 45)
        #p1.hbar(y = 'Rank', height = 0.9, right = 'Count', source = ColumnDataSource(data_text), color = "maroon")
        p1.xaxis[0].ticker.desired_num_ticks = 10
        p1.xaxis.major_label_overrides = {1: list(ss.Word)[0],
                                2: list(ss.Word)[1],
                                3: list(ss.Word)[2],
                                4: list(ss.Word)[3],
                                5: list(ss.Word)[4],
                                6: list(ss.Word)[5],
                                7: list(ss.Word)[6],
                                8: list(ss.Word)[7],
                                9: list(ss.Word)[8],
                                10: list(ss.Word)[9]}
        p1.vbar(x="Rank", top="Count", width = 0.95, source = ColumnDataSource(data_text), color = "color")
        #p1.add_layout(labels_text)
        p1.xaxis.minor_tick_line_color = None
        p1.xaxis.major_tick_line_color = None
        p1.xgrid.visible = False
        p1.ygrid.visible = False
        p1.add_tools(hover1)
        
        
        ###Actual wordcloud
        from nltk.corpus import stopwords
        stopwords = set(STOPWORDS)
        stopwords.add('one')
        stopwords.add('also')
        stopwords.add('twitter')
        stopwords.add('pic')
        stopwords.add('https')
        stopwords.add('bit')
        stopwords.add('ly')
        stopwords.add('via')
        stopwords.add('buff')
        wordcloud_good = WordCloud(colormap = "Dark2", width = 750, height = 500, stopwords=stopwords, scale = 4, max_words = 300,
                          background_color='white').generate(combined_tweets)
        wordcloud_good.to_file('temp.png')
        word_img = Image.open('temp.png').convert('RGBA')
        xdim, ydim = word_img.size
        img = np.empty((ydim, xdim), dtype=np.uint32)
        view = img.view(dtype=np.uint8).reshape((ydim, xdim, 4))
        view[:,:,:] = np.flipud(np.asarray(word_img))

        dim = max(xdim, ydim)
        fig.image_rgba(image=[img], x=0, y=0, dw=500, dh=750)
        
        
        ##Sentiments graph
        def loadLexicon(fname):
            newLex=set()
            lex_conn=open(fname)
            #add every word in the file to the set
            for line in lex_conn:
                newLex.add(line.strip())# remember to strip to remove the lin-change character
            lex_conn.close()
            return newLex
        sia = SIA()
        results = []
        for line in alltweets:
            pol_score = sia.polarity_scores(line)
            pol_score['Tweet'] = line
            results.append(pol_score)
        polarity = pd.DataFrame(results)
        most_neg = polarity.sort_values('compound', ascending = True)[0:5]
        most_pos = polarity.sort_values('compound', ascending = False)[0:5]
        polarity['Polarity'] = 0
        polarity.loc[polarity['compound'] > 0, 'Polarity'] = 1
        polarity.loc[polarity['compound'] < 0, 'Polarity'] = -1
        perc = pd.DataFrame(polarity.Polarity.value_counts(normalize=True) * 100).reset_index().rename(columns = {'index':'Sentiment',
                                                                                                                  'Polarity':'Percentage'})
        perc.Percentage = perc.Percentage.apply(lambda x : round(x, 2))
        perc["Percentage_Text"] = perc.Percentage.apply(lambda x : str(round(x, 2)) + "%")
        data_text = dict(Sentiment = list(perc.Sentiment), Percentage = list(perc.Percentage), color = Category20[3],
                         Percentage_Text = list(perc.Percentage_Text))
        labels = LabelSet(x="Sentiment", y="Percentage", text="Percentage_Text", level='glyph', render_mode='css', source = ColumnDataSource(data_text),
                          x_offset = -10, text_font_size="10pt")
        sent.xaxis[0].ticker.desired_num_ticks = 3
        sent.xaxis.major_label_overrides = {1: 'Positive',
                                         0: 'Neutral',
                                         -1: 'Negative'}
        sent.xaxis.minor_tick_line_color = None
        sent.xaxis.major_tick_line_color = None
        sent.vbar(x="Sentiment", top="Percentage", width = 0.80, source = ColumnDataSource(data_text), color = "color")
        sent.xgrid.visible = False
        sent.ygrid.visible = False
        sent.add_layout(labels)
        
        ##Emotion break down
        nrc = pd.read_csv("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", sep = "\t", header = None, names=["term", "category", "flag"])
        emotions = []
        final = pd.DataFrame()
        term = list(nrc.term)
        for i in range(len(cleaned_tweets)):
            if cleaned_tweets[i] in term:
                sub = nrc[nrc.term == cleaned_tweets[i]]
                s = sub[sub.flag == 1]
                if list(s.category) != []:
                    emotions.append(list(s.category))
        emotions_clubed = []
        for i in emotions:
            for j in range(len(i)):
                if i[j] == 'positive':
                    i[j] = 'joy'
                elif i[j] == 'negative':
                    i[j] = 'sadness'
                emotions_clubed.append(i[j])
        radar = pd.DataFrame(pd.Series(emotions_clubed))[0].value_counts(normalize = True)
        radar = round(radar * 100 , 2)
        radar_df = pd.DataFrame(radar).reset_index()
        radar_df['angle'] = radar_df[0]/radar_df[0].sum() * 2*pi
        radar_df['color'] = Category20c[radar_df.shape[0]]
        radar_df = radar_df.rename(columns = {'index':'Emotion', 0:'Percentage', 'angle':'Angle', 'color':'Color'})
        source_events = ColumnDataSource(data=dict(Emotion = radar_df['Emotion'], Percentage = radar_df['Percentage'],
                                           Angle = radar_df['Angle'], Color = radar_df['Color']))
        plot_events.wedge(x=0, y=1, radius=0.47, start_angle=cumsum('Angle', include_zero=True), end_angle=cumsum('Angle'),
                  line_color="white", fill_color='Color', source=source_events, legend = "Emotion")
        
        plot_events.add_tools(hover2)
        
        
        ##LDA
        stopwords = set(STOPWORDS)
        stopwords.add('one')
        stopwords.add('also')
        stopwords.add('twitter')
        stopwords.add('pic')
        stopwords.add('https')
        stopwords.add('bit')
        stopwords.add('ly')
        stopwords.add('via')
        stopwords.add('buff')
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=stopwords)
        matrix = tf_vectorizer.fit_transform(alltweets)
        vocab=tf_vectorizer.get_feature_names()
        model = lda.LDA(n_topics=topic_num, n_iter=900)
        model.fit(matrix)
        topics = []
        topics.append("<b>Most Frequent Words used in 12 different Topics found in the search</b><br><br>")
        top_words_num=20
        topic_mixes= model.topic_word_
        for i in range(topic_num):#for each topic
            top_indexes=np.argsort(topic_mixes[i])[::-1][:top_words_num]                              
            my_top=''
            for ind in top_indexes:my_top+=vocab[ind]+' '
            topics.append('TOPIC:'+str(i+1)+' --> '+str(my_top)+'<br><br>')
        pos_nes = []
        pos_neg = []
        pos_nes.append("<b> Top 5 most Positive Tweets </b><br><br>")
        pos = (["> "+i+'<br><br>' for i in most_pos['Tweet']])
        pos_nes.append(pos)
        pos_nes.append("<b> Top 5 most Negative Tweets </b><br><br>")
        neg = (["> "+i+'<br><br>' for i in most_neg['Tweet']])
        pos_nes.append(neg)
        for i in range(len(pos_nes)):
            pos_neg.append("".join(pos_nes[i]))
        d = Div(text = """<div style="width: 49%; text-align: justify; float: left">"""+"".join(pos_neg)+"</div>"+"""
        <div style="width: 49%; text-align: justify; float: right">"""+"".join(topics)+"""
        </div><div style="width: 2%;text-align:justify;float:center">"""+" "+"</div>", width = 1500, height = 500)
        column2.children.append(d)
        
    else:
        pass

    
button.on_click(on_buttonpress)

#div1 = Div(text="""<img width = "1470" height = "80" src="https://drive.google.com/open?id=1efVHKDtRkSM0KO0kQ8A8I0YuSknNsaHY">""")


#div1 = Div(text = """<img width = "1470" height = "80" src = "F:/PRISMOJI/Emoji App/myapp/static/images/twitter-header.jpg" class="custom-logo" alt="Twitter #Logo" itemprop="logo">""")

div1 = Div(text = """<html>
   <head>
      <style type="text/css">
         body {
            padding: 0in; 
         }
         .square {
         position: relative;
            background-color: #38A1F3;
            width: 1500px;
            height: 80px;
            border: 0px;
         }
         .square p{
          font-size:30pt;
          margin: 0;
          background: #38A1F3;
          position: relative;
          top: 20%;
          left: -17%;
          text-align: center;
          margin:auto;
          padding:0;
          height:5%;
          font-family:Helvetica;
          color:white;
      width:35%;
      transform: translate(50%, 50%)   
         }
      </style>
   </head>
   <body>
      <div class="square">
      <p>Twitter Analytics Dashboard</p></div>
   </body>
</html>""")

layout = column(column(row(div1), row(widgetbox(text_input, width = 220), column(Div(text = ""), buttoncomb)),
                                        row(p, fig), row(sent, plot_events), column(Div(text = ""), column2)))
curdoc().add_root(layout)
curdoc().title = "Twitter Analytics App"