#module 1
import re
import tweepy,csv
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords 
from textblob import TextBlob
#from textblob import TextBlob
#import matplotlib.pyplot as plt

consumerKey = 'gosh23ILRVeTZrvNRN9zMO2qp'
consumerSecret = 'tMDRVUF0hG4ADclHzMhZG3Sm2kQBvTtek9Z2oQAKAuOjfrFd1F'
accessToken = '849664475984797696-gDdD9zEEPXUiIWlkidLEn3mcsZvD0uI'
accessTokenSecret = 'iJDNQv39nrELrhdXVDCXe4VXLEF2lYyJu02cLBoEoQipY'
auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)

searchTerm = input("Enter Product/Brand to search about: ")
NoOfTerms = int(input("Enter how many tweets to search: "))

Total=NoOfTerms

tweets=[]
tweets = tweepy.Cursor(api.search, q=searchTerm, lang = "en").items(NoOfTerms)

  # Open/create a file to append data to
csvFile = open('searchresult.csv', 'a')
csvWriter = csv.writer(csvFile)

tweetText=[]

for tweet in tweets:
            #Append to temp so that we can store in csv later. I use encode UTF-8
            tweetText.append(tweet.text.encode('utf-8'))

csvWriter.writerow(tweetText)
csvFile.close()

#module 2

stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])
    
csvFile1 = open('preprocessed.csv', 'a')
csvWriter1 = csv.writer(csvFile1)

intermediate=[]
processedTweets=[]  #list of words
for tweet in tweetText:

    tweet = tweet.lower() # convert text to lower-case
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet.decode('utf-8')) # remove URLs
    tweet = re.sub('@[^\s]+' ,'AT_USER', tweet) # remove usernames
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
    #print(tweet)
    intermediate.append(tweet)
    tweet = word_tokenize(tweet)# remove repeated characters (helloooooooo into hello)
    for word in tweet:
        if word not in stopwords:
            processedTweets.append(word)
            
csvWriter1.writerow(intermediate)
csvFile1.close()
    
#print(processedTweets)

#module 3

wordfreq=[]
for w in processedTweets:
    wordfreq.append(processedTweets.count(w))

#print("Frequency\n" + str(list(zip(processedTweets, wordfreq))))

from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt

comment_words=''
stopwords=set(STOPWORDS)

textdata=''
for i in processedTweets:
    textdata+=i
    textdata+=' '
    
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white',
                min_font_size = 10).generate(textdata) 

wordcloud.to_file('Wordcloud.png')

plt.title(searchTerm + 'WordCloud')
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 



#module 4

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(processedTweets)

true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    arr=[]
    for ind in order_centroids[i, :10]:
        #print(' %s' % terms[ind])
        arr.append(terms[ind])
    print(arr)
    
#moudle 5
    
polarity = 0
positive = 0
wpositive = 0
spositive = 0
negative = 0
wnegative = 0
snegative = 0
neutral = 0

NoOfTerms=0
for tweet in processedTweets:
    analysis = TextBlob(tweet)
    #print(analysis.sentiment)  # print tweet's polarity
    polarity += analysis.sentiment.polarity  # adding up polarities to find the average later
    NoOfTerms+=1
    if (analysis.sentiment.polarity == 0):  # adding reaction of how people are reacting to find average later
        neutral += 1
    elif (analysis.sentiment.polarity > 0 and analysis.sentiment.polarity <= 0.3):
        wpositive += 1
    elif (analysis.sentiment.polarity > 0.3 and analysis.sentiment.polarity <= 0.6):
        positive += 1
    elif (analysis.sentiment.polarity > 0.6 and analysis.sentiment.polarity <= 1):
        spositive += 1
    elif (analysis.sentiment.polarity > -0.3 and analysis.sentiment.polarity <= 0):
        wnegative += 1
    elif (analysis.sentiment.polarity > -0.6 and analysis.sentiment.polarity <= -0.3):
        negative += 1
    elif (analysis.sentiment.polarity > -1 and analysis.sentiment.polarity <= -0.6):
        snegative += 1


#print(positive,spositive,wpositive,neutral,negative,wnegative,snegative)

# function to calculate percentage
def percentage(part, whole):
    temp = 100 * float(part) / float(whole)
    return format(temp, '.2f')

def plotPieChart(positive, wpositive, spositive, negative, wnegative, snegative, neutral, searchTerm, noOfSearchTerms):
        labels = ['Positive [' + str(positive) + '%]', 'Weakly Positive [' + str(wpositive) + '%]','Strongly Positive [' + str(spositive) + '%]', 'Neutral [' + str(neutral) + '%]',
                  'Negative [' + str(negative) + '%]', 'Weakly Negative [' + str(wnegative) + '%]', 'Strongly Negative [' + str(snegative) + '%]']
        sizes = [positive, wpositive, spositive, neutral, negative, wnegative, snegative]
        colors = ['yellowgreen','lightgreen','darkgreen', 'gold', 'red','lightsalmon','darkred']
        patches, texts = plt.pie(sizes, colors=colors, startangle=90)
        plt.legend(patches, labels, loc="best")
        plt.title('How people are reacting on ' + searchTerm + ' by analyzing ' + str(noOfSearchTerms) + ' Tweets.')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

        
 # finding average of how people are reacting
positive =percentage(positive, NoOfTerms)
wpositive = percentage(wpositive, NoOfTerms)
spositive = percentage(spositive, NoOfTerms)
negative = percentage(negative, NoOfTerms)
wnegative = percentage(wnegative, NoOfTerms)
snegative =percentage(snegative, NoOfTerms)
neutral = percentage(neutral, NoOfTerms)

# finding average reaction
polarity = polarity / NoOfTerms

 # printing out data
print("How people are reacting on " + searchTerm + " by analyzing " + str(Total) + " tweets.")
print()
print("General Report: ",end='')

if (polarity == 0):
    print("Neutral")
elif (polarity > 0 and polarity <= 0.3):
    print("Weakly Positive")
elif (polarity > 0.3 and polarity <= 0.6):
    print("Positive")
elif (polarity > 0.6 and polarity <= 1):
    print("Strongly Positive")
elif (polarity > -0.3 and polarity <= 0):
    print("Weakly Negative")
elif (polarity > -0.6 and polarity <= -0.3):
    print("Negative")
elif (polarity > -1 and polarity <= -0.6):
    print("Strongly Negative")
    
plotPieChart(positive, wpositive, spositive, negative, wnegative, snegative, neutral, searchTerm, NoOfTerms)
