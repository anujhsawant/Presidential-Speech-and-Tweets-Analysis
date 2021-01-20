import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import string
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
from collections import Counter
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import pyLDAvis.sklearn
import concurrent.futures
from datetime import datetime
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer

#Defining all the functions

def opentxtfile(filename):
    with open(filename,'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        transcripts=''.join(lines)
        
    return transcripts


def ngram(text,grams):    
    n_grams_list = []    
    count = 0    
    for token in text[:len(text)-grams+1]:       
        #n_grams_list.append(text[count:count+grams])      
        n_grams_list.append(text[count]+' '+text[count+grams-1])        
        count=count+1      
    return n_grams_list


def most_common(lst, num):
    
    data = Counter(lst)    
    common = data.most_common(num)    
    top_comm = []    
    for i in range (0, num):        
        top_comm.append (common [i][0])    
    return top_comm



def word_cloud(name,transcript,color_map,x):
    wc = WordCloud(stopwords=stop_words, width = 300, height = 250,background_color="white", colormap=color_map,
                   max_font_size=75, random_state=60,max_words=2000)
    plt.rcParams['figure.figsize'] = [x,x]
    wc.generate(transcript)
    plt.subplot(3,4,3)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(name)
    plt.show()



def split_text(text, n=10):
    '''Takes in a string of text and splits into n equal parts, with a default of 10 equal parts.'''

    # Calculate length of text, the size of each chunk of text and the starting points of each chunk of text
    length = len(text)
    size = math.floor(length / n)
    start = np.arange(0, length, size)
    
    # Pull out equally sized pieces of text and put it into a list
    split_list = []
    for piece in range(n):
        split_list.append(text[start[piece]:start[piece]+size])
    return split_list


def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    #text = " ".join(text)
    text = text.lower()
    text = text.strip()
    text = re.sub('\[.*?\]’', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = text.replace("’","")
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
        
    return text

#Adding more content based stop words to the list

more_stp_words = ['mr','mrs','an','am','said','dont','like','dint','know','ive','got'
                 ,'could','see','havent','im','going','looked','through','tell','yeh',
                 'go','back','get','yer','would','never','seen','something','else','next','day','years'
                 ,'didnt','look','one','ever','even','though','although','every','time','make','sure'
                 ,'told','minutes','minute','hour','where','else','no','oh','ten','year','nine',
                 'ten','points','fifty','hundred','twenty','gotten','fell','asleep','yes','trying','find','one','two','three'
                 ,'clock','give','us','youd','expect','picked','lived','number','twice','youknowwho','think',
                  'well','around','new','let','need','want','thing','also','tonight','must','american','america','take','last','many',
                 'say','come','thats','put','americans','right','united states','united','states','really','still','working','work']
stop_words.extend(more_stp_words)


def final_clean(texts):
    toks = word_tokenize(texts)
    stp = [word for word in toks if word not in stop_words]
    #stpr = ' '.join(stp)
    return stp


#Reading all the text files 

BarackObama_transcripts=opentxtfile('BO.txt')
DonaldTrump_transcripts=opentxtfile('DT.txt')

BO_Deathof_transcript=opentxtfile('BO_Death of Osama Bin Laden.txt')
DT_Deathof_transcript=opentxtfile('DT_Death of Abu Bakr al-Baghdadi.txt')

BO_SchoolShooting_transcript=opentxtfile('BO_School Shooting.txt')
DT_SchoolShooting_transcript=opentxtfile('DT_School Shooting.txt')

BO_Victory_transcript=opentxtfile('BO_Victory Speech.txt')
DT_Victory_transcript=opentxtfile('DT_Victory Speech.txt')

BO_StateUnionAddress_transcript=opentxtfile('BO_State of the Union Address.txt')
DT_StateUnionAddress_transcript=opentxtfile('DT_State of the Union Address.txt')

BO_IntroVP_transcript=opentxtfile('BO_Introducing VP.txt')
DT_IntroVP_transcript=opentxtfile('DT_Introducing VP.txt')

BO_InauguralSpeech_transcript=opentxtfile('BO_Inaugural Speech.txt')
DT_InauguralSpeech_transcript=opentxtfile('DT_Inaugural Speech.txt')

BO_Immigration_transcript=opentxtfile('BO_Immigration.txt')
DT_Immigration_transcript=opentxtfile('DT_Immigration.txt')

BO_Christmas_transcript=opentxtfile('BO_Christmas.txt')
DT_Christmas_transcript=opentxtfile('DT_Christmas.txt')

BO_AnnualPrayerBreakfast_transcript=opentxtfile('BO_Annual National Prayer Breakfast.txt')
DT_AnnualPrayerBreakfast_transcript=opentxtfile('DT_Annual National Prayer Breakfast.txt')

BO_AddressCongress_transcript=opentxtfile('BO_Address to Joint Session of Congress.txt')
DT_AddressCongress_transcript=opentxtfile('DT_Address to Joint Session of Congress.txt')


#Removing unnecessary characters (text cleaning)

BO_cleaned_transcripts=clean_text(BarackObama_transcripts)
DT_cleaned_transcripts=clean_text(DonaldTrump_transcripts)


#Tokenizing and removing stopwords

BO_cleaned_transcripts=final_clean(BO_cleaned_transcripts)
DT_cleaned_transcripts=final_clean(DT_cleaned_transcripts)

BO_cleaned_transcripts_1 = ' '.join(BO_cleaned_transcripts)
DT_cleaned_transcripts_1 = ' '.join(DT_cleaned_transcripts)


#Creating a dataframe 


df_transcripts=pd.DataFrame({"Transcripts" :(BO_cleaned_transcripts_1,DT_cleaned_transcripts_1) },index= ("Barack","Donald"))
df_transcripts


#Creating a document term matrix 

cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(df_transcripts.Transcripts)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = df_transcripts.index
data_dtm


#Transposing the document term matrix


data = data_dtm
data = data.transpose()


#Top words for both the Presidents 


top_words=[]
top_dict = {}
for c in data.columns:
    top = data[c].sort_values(ascending=False).head(10)
    top_dict[c]= list(zip(top.index, top.values))

top_dict


#Plotting the wordclouds


word_cloud('Donald Trump',DT_cleaned_transcripts_1,'autumn',50)
word_cloud('Barack Obama',BO_cleaned_transcripts_1,'winter',50)


#Vocabulary of both Presidents


unique_list = []
full_names=['Barack','Donald']
for president in data.columns:
    uniques = data[president].to_numpy().nonzero()[0].size
    unique_list.append(uniques)
    

# Create a new dataframe that contains this unique word count
data_words = pd.DataFrame(list(zip(full_names, unique_list)), columns=['President', 'Total Unique words'])

BO_Deathof_transcript=clean_text(BO_Deathof_transcript)
DT_Deathof_transcript=clean_text(DT_Deathof_transcript)

df_Deathof_transcript=pd.DataFrame({'Transcripts': [BO_Deathof_transcript,DT_Deathof_transcript]},index=['Barack', 'Donald'])
df_Deathof_transcript

#Calculating the Polarity and subjectivity of both the Presidents

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity


df_Deathof_transcript['Polarity'] = df_Deathof_transcript['Transcripts'].apply(pol)
df_Deathof_transcript['Subjectivity'] = df_Deathof_transcript['Transcripts'].apply(sub)
df_Deathof_transcript



def plot_fn(df_transcript,measure):
    fig = plt.figure(figsize=(30, 2))

    # draw lines
    xmin = -1
    xmax = 1
    y = 5
    height = 2

    plt.hlines(y, xmin, xmax)
    plt.vlines(xmin, y - height / 2., y + height / 2.)
    plt.vlines(xmax, y - height / 2., y + height / 2.)

    # draw a point on the line
    px1 = df_Deathof_transcript[measure]['Barack']
    plt.plot(px1,y, 'bo', ms = 30, mfc = 'b')

    px2 = df_Deathof_transcript[measure]['Donald']
    plt.plot(px2,y, 'ro', ms = 30, mfc = 'r')
    
    
    plt.title(measure, loc='center', pad=50,fontsize=50)
    plt.text(px1,y+1,round(px1,2),horizontalalignment='center',verticalalignment='center',fontsize=25)
    plt.text(px2,y+1,round(px2,2),horizontalalignment='center',verticalalignment='center',fontsize=25)

    # add numbers
    if(measure=='Polarity'):
        plt.text(xmin - 0.01, y, 'Negative(-1)', horizontalalignment='right',verticalalignment='center',fontsize=30)
       
        plt.text(px1,y-1.5,'Barack Obama',horizontalalignment='left',verticalalignment='center',fontsize=25)
        plt.text(px2,y-1.5,'Donald Trump',horizontalalignment='right',verticalalignment='center',fontsize=25)
        plt.text(xmax + 0.01, y, 'Positive(+1)', horizontalalignment='left',verticalalignment='center',fontsize=30)
       
    else:
        plt.text(xmin - 0.01, y, 'Objective(0)', horizontalalignment='right',verticalalignment='center',fontsize=30)
        plt.text(xmin-0.1 , y-1.5, 'Fact', horizontalalignment='left',verticalalignment='center',fontsize=30)

        plt.text(px1,y-1.5,'Barack Obama',horizontalalignment='right',verticalalignment='center',fontsize=25)
        plt.text(px2,y-1.5,'Donald Trump',horizontalalignment='left',verticalalignment='center',fontsize=25)
        plt.text(xmax + 0.01, y, 'Subjective(1)', horizontalalignment='left',verticalalignment='center',fontsize=30)
        plt.text(xmax + 0.04, y-1.5, 'Opinion', horizontalalignment='left',verticalalignment='center',fontsize=30)
       

    plt.axis('off')
    plt.show()

#Plotting the Polarity and Subjectivity

plot_fn(df_Deathof_transcript,'Polarity')
plot_fn(df_Deathof_transcript,'Subjectivity')


data = {'Name':['Barack', 'Donald','Barack', 'Donald','Barack', 'Donald','Barack', 'Donald','Barack', 'Donald','Barack', 'Donald','Barack', 'Donald','Barack', 'Donald','Barack', 'Donald','Barack', 'Donald'],'Event' : ['School Shooting','School Shooting','Victory','Victory','State of the Union Address','State of the Union Address','Introducing VP Joe Biden','Introducing VP Mike Pence','Inaugural Speech','Inaugural Speech','Immigration','Immigration,','Christmas','Christmas','Annual National Prayer Breakfast','Annual National Prayer Breakfast','Address to Joint Session of Congress','Address to Joint Session of Congress','Death of Osama Bin Laden ','Death of Abu Bakr Al-Baghdadi'], 'Transcripts':[BO_SchoolShooting_transcript,DT_SchoolShooting_transcript,BO_Victory_transcript,DT_Victory_transcript,BO_StateUnionAddress_transcript,DT_StateUnionAddress_transcript,BO_IntroVP_transcript,DT_IntroVP_transcript,BO_InauguralSpeech_transcript,DT_InauguralSpeech_transcript,BO_Immigration_transcript,DT_Immigration_transcript,BO_Christmas_transcript,DT_Christmas_transcript,BO_AnnualPrayerBreakfast_transcript,DT_AnnualPrayerBreakfast_transcript,BO_AddressCongress_transcript,DT_AddressCongress_transcript,BO_Deathof_transcript,DT_Deathof_transcript]}

df_all_transcripts = pd.DataFrame(data) 
df_all_transcripts.head()

#Cleaning the transcripts

for i in range(len(df_all_transcripts)):
    df_all_transcripts['Transcripts'][i]=clean_text(df_all_transcripts['Transcripts'][i])
df_all_transcripts.head()



BO_split=[]
DT_split=[]

for i in range(0,20,2):
    BO_split.append(split_text(df_all_transcripts.iat[i,2]))
for i in range(1,20,2):
    DT_split.append(split_text(df_all_transcripts.iat[i,2]))

BO_polarity=[]
DT_polarity=[]

for i in range(10):
    for j in range(10):
        BO_polarity.append(TextBlob(BO_split[i][j]).sentiment.polarity)
        DT_polarity.append(TextBlob(DT_split[i][j]).sentiment.polarity)


plt.rcParams['figure.figsize'] = [20, 16]
n=0
nw=0
for nr in range(0,100,10):    
    plt.subplot(3, 4, n+1)
    n+=1
    plt.plot(BO_polarity[nr:nr+10])
    plt.plot(np.arange(0,10), np.zeros(10),color='grey')
    plt.title("BO:"+df_all_transcripts.iat[nw+1,1])
    
    plt.plot(DT_polarity[nr:nr+10],color='red')
    #plt.plot(np.arange(0,10), np.zeros(10),color='grey')
    plt.title("DT:"+df_all_transcripts.iat[nw,1]+"\nBO:"+df_all_transcripts.iat[nw+1,1])
   
    nw+=2
    
    plt.ylim(ymin=-.4, ymax=.8)
    
plt.show()



#Analyzing Trump tweets (Part 2)


#Reading the csv file with trump tweets
df = pd.read_excel('Trump_tweets.xlsx')
df.head()

#the 10 most popular hashtags words

popular_hashtags = Counter()
for x in df['text']:
    for line in x.split():
        if '#' in line:
            popular_hashtags[line] += 1
print('The top 10 Hastags President trump mentioned: \n\n',popular_hashtags.most_common(10))

#the 5 most cited screen-names

screen_names = Counter()
for x in df['text']:
    for line in x.split():
        if '@' in line:
            screen_names[line] += 1
print('The top 10 people President trump tagged/mentioned: \n\n',screen_names.most_common(10))

#Visualizing hourly tweets according to most frequest keywords

df['hour'] = df['created_at'].dt.hour

df.head()

tweetsPerHour = df[['hour', 'text']].pivot_table(
    index='hour', aggfunc='count')

tweetsPerHour.iloc[::-1].plot(kind='barh', stacked=True)

from IPython.display import display, Markdown

def analyzeText(search):
    display(Markdown('## Search: _{0}_'.format(search)))
    filtered = df[df['text'].str.contains(search, case=False)]
    count = len(filtered)
    percent = float(len(filtered))/len(df) * 100
    display(Markdown('Number of Tweets: {0}'.format(count)))
    display(Markdown('Percent of all Tweets: {0:.2f}%'.format(percent)))

    # set up the filtered data frame
    
    analyzedByHour = filtered[['hour', 'text']].pivot_table(index=['hour'], aggfunc='count')

    # adjust the filtered data frame to use the index from the base data frame
    
    analyzedByHour = analyzedByHour.reindex(range(0, 24)).fillna(0)


    analyzedByHourPlot = analyzedByHour.plot(kind='bar', stacked=True, title='{0} by hour of day'.format(search))
    plt.show()

searches = ['fake media', 'complete endorsement', 'maga', 'radical left', 'russia' ]
for search in searches:
    analyzeText(search)

#Creating a list of all the tweets

tweets = []
for i in df['text']:
    tweets.append(i)

tweets = ' '.join(tweets)

for i in range(0,len(df)):
    df['text'][i] = clean_text(df['text'][i])


tweets1 = clean_text(tweets)

more_stp_words = ['rt','trump','donald','amp','us','realdonaldtrump','new','get','would']
stop_words.extend(more_stp_words)

tweets2 = final_clean(tweets1)

bi_grams = ngram(tweets2,2)

#The top 10 most common bigrams are 
most_common(bi_grams,10)


def selected_topics(model, vectorizer, top_n=10):    
    for idx, topic in enumerate(model.components_):        
        print("Topic %d:" % (idx))        
        print([(vectorizer.get_feature_names()[i], topic[i])                        
               for i in topic.argsort()[:-top_n - 1:-1]])
        print('\n')


vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words=stop_words, lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(df['text'])

lda = LatentDirichletAllocation(n_components=5, max_iter=10, learning_method='online',verbose=False)
data_lda = lda.fit_transform(data_vectorized)

# Keywords for topics clustered by Latent Dirichlet Allocation
print('##############THE 5 TOPICS IDENTIFIED IN PRESIDENT TRUMPS TWEETS ARE AS FOLLOWS###############')
print('\nLDA Model:')
selected_topics(lda, vectorizer)
# visualizing the results. An html interactive file will be created
dash = pyLDAvis.sklearn.prepare(lda, data_vectorized, vectorizer, mds='tsne')
pyLDAvis.save_html(dash, 'LDA_Visualization.html')


def get_NER(text_data):
    #/Users/prajwalshreyas/Desktop/Singularity/dockerApps/ner-algo/stanford-ner-2015-01-30
    stanford_classifier = ('/Users/akashbhoite/Downloads/stanford-ner-2020-11-17/classifiers/english.all.3class.distsim.crf.ser.gz')
    stanford_ner_path = ('/Users/akashbhoite/Downloads/stanford-ner-2020-11-17//stanford-ner.jar')

    #try:
        # Creating Tagger Object
    st = StanfordNERTagger(stanford_classifier, stanford_ner_path, encoding='utf-8')
    #except Exception as e:
    #       print (e)

    # Get keyword for the input data frame
    #keyword = tweetDataFrame.keyword.unique()
    # Subset column containing tweet text and convert to list
    # next insert a placeholder ' 12345678 ' to signify end of individual tweets

    #text_data = pd.read_json('/Users/prajwalshreyas/Desktop/Singularity/dockerApps/sentiment-algo/app-sentiment-algo/sample_text.json')
    print ('start get_NER')
    text_out = text_data.copy()
    doc = [ docs + ' 12345678 ' for docs in list(text_data['text'])]
    # ------------------------- Stanford Named Entity Recognition
    tokens = nltk.word_tokenize(str(doc))
    entities = st.tag(tokens) # actual tagging takes place using Stanford NER algorithm


    entities = [list(elem) for elem in entities] # Convert list of tuples to list of list
    print ('tag complete')
    for idx,element in enumerate(entities):
        try:
            if entities[idx][0] == '12345678':
                entities[idx][1] = "DOC_NUMBER"  #  Modify data by adding the tag "Doc_Number"
            #elif entities[idx][0].lower() == keyword:
            #    entities[idx][1] = "KEYWORD"
            # Combine First and Last name into a single word
            elif entities[idx][1] == "PERSON" and entities[idx + 1][1] == "PERSON":
                entities[idx + 1][0] = entities[idx][0] + '-' + entities[idx+1][0]
                entities[idx][1] = 'Combined'
            # Combine consecutive Organization names
            elif entities[idx][1] == 'ORGANIZATION' and entities[idx + 1][1] == 'ORGANIZATION':
                entities[idx + 1][0] = entities[idx][0] + '-' + entities[idx+1][0]
                entities[idx][1] = 'Combined'
        except IndexError:
            break
    print ('enumerate complete')
    # Filter list of list for the words we are interested in
    filter_list = ['DOC_NUMBER','PERSON','LOCATION','ORGANIZATION']
    entityWordList = [element for element in entities if any(i in element for i in filter_list)]

    entityString = ' '.join(str(word) for insideList in entityWordList for word in insideList) # convert list to string and concatenate it
    entitySubString = entityString.split("DOC_NUMBER") # split the string using the separator 'TWEET_NUMBER'
    del entitySubString[-1] # delete the extra blank row created in the previous step

    # Store the classified NERs in the main tweet data frame
    for idx,docNER in enumerate(entitySubString):
        docNER = docNER.strip().split() # split the string into word list
        # Filter for words tagged as Organization and store it in data frame
        text_out.loc[idx,'Organization'] =  ','.join([docNER[i-1]  for i,x in enumerate(docNER) if x == 'ORGANIZATION'])
        # Filter for words tagged as LOCATION and store it in data frame
        text_out.loc[idx,'Place'] = ','.join([docNER[i-1] for i,x in enumerate(docNER) if x == 'LOCATION'])
        # Filter for words tagged as PERSON and store it in data frame
        text_out.loc[idx,'Person'] = ','.join([docNER[i-1]  for i,x in enumerate(docNER) if x == 'PERSON'])

    print ('process complete')
    return text_out


text_ner_out = get_NER(df)

#the outputs of the ner tagger
NER = text_ner_out.loc[(text_ner_out['Place'] != '') | (text_ner_out['Organization'] != '')|(text_ner_out['Person'] != '')][['text','Organization','Place','Person']].head(30)

#Printing the output of NER
NER.head(20)





