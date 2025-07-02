import json
import datetime as dt
import os
from typing import List, Dict, TypedDict, Optional
from praw import Reddit
from praw.models import Submission
from dotenv import load_dotenv

load_dotenv()

class Comment(TypedDict):
    author: Optional[str]
    body: str
    score: int
    created: float
    created_date: str

class Post(TypedDict):
    id: str
    title: str
    url: str
    created: float
    created_date: str
    subreddit: str
    body: str
    score: int
    comments: List[Comment]

class RedditService:
    def __init__(self, user_agent: str = "trading-llm:v1.0"):
        client_id = os.getenv("REDDIT_APP_ID")
        client_secret = os.getenv("REDDIT_APP_KEY")
        
        if not client_id or not client_secret:
            raise ValueError(
                "Reddit API credentials not found in environment variables. "
                "Please set REDDIT_APP_ID and REDDIT_APP_KEY"
            )
            
        self.reddit = Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )

    def get_posts_with_top_comments(
        self,
        subreddit_name: str,
        posts_limit: int = 100,
        comments_limit: int = 20
    ) -> List[Post]:
        all_posts = []
        subreddit = self.reddit.subreddit(subreddit_name)
        for submission in subreddit.new(limit=posts_limit):
            post_date = dt.datetime.fromtimestamp(submission.created_utc)
            
            # Set sort to 'top' and fetch comments
            submission.comment_sort = "top"
            submission.comments.replace_more(limit=0)
            # Get top N comments by score
            top_comments = sorted(
                submission.comments.list(),
                key=lambda c: c.score if hasattr(c, "score") else 0,
                reverse=True
            )[:comments_limit]
            
            post_record: Post = {
                "id": submission.id,
                "title": submission.title,
                "url": submission.url,
                "created": submission.created_utc,
                "created_date": post_date.isoformat(),
                "subreddit": subreddit_name,
                "body": submission.selftext,
                "score": submission.score,
                "comments": [
                    {
                        "author": c.author.name if c.author else None,
                        "body": c.body,
                        "score": c.score,
                        "created": c.created_utc,
                        "created_date": dt.datetime.fromtimestamp(c.created_utc).isoformat()
                    }
                    for c in top_comments
                ]
            }
            all_posts.append(post_record)
        return all_posts

