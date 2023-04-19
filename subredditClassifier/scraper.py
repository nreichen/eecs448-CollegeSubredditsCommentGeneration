import sys
import praw
from praw.models import MoreComments
import pandas as pd
import re
import string
from tqdm import tqdm

subreddits = [
    "AskReddit",
    "gaming",
    "Music",
    "todayilearned",
    "movies",
    "Showerthoughts",
    "askscience",
    "books",
    "explainlikeimfive",
    "LifeProTips"
]
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\s+',' ',text)
    return text.strip()

reddit = praw.Reddit(client_id='pT2OJLIzhvLJK_qlz2nG2A', 
                        client_secret='7J4qmfY6-9IK56TR2UJxXfmL6mcLig', 
                        user_agent='Comment Scraper')

if len(sys.argv) == 2 and sys.argv[-1] == '-v':
    with open('new_posts.csv', 'w') as file:
        for s in subreddits:
            subreddit = reddit.subreddit(s)
            for post in subreddit.hot(limit=1000):
                file.write(str(post.id) + ',' + str(s) + '\n')

if len(sys.argv) == 2 and sys.argv[-1] == '-c':
    with open('new_comments.csv', 'w') as file:
        df = pd.read_csv('posts.csv', names=['id', 'subreddit'])
        for _, row in tqdm(df.iterrows()):
            submission = reddit.submission(id=row['id'])
            submission.comments.replace_more(limit=0)
            for comment in submission.comments:
                file.write(str(clean_text(comment.body)) + ',' + str(row['subreddit']) + '\n')

if len(sys.argv) == 2 and sys.argv[-1] == '-s':
    posts = pd.read_csv('new_posts.csv', names=['id', 'subreddit'])
    comments = pd.read_csv('new_comments.csv', names=['comment', 'subreddit'])
    print(posts.describe())
    print(comments.describe())
    for s in subreddits:
        print(s + '\t' + str(len(comments[comments['subreddit']==s])))



