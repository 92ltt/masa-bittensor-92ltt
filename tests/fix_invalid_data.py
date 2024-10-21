import bittensor as bt
import re
from typing import List, Optional, Any, Dict
from datetime import datetime, UTC
import json
import torch
from sentence_transformers import SentenceTransformer
from masa.types.twitter import ProtocolTwitterTweetResponse
import requests

class TwitterTweetsRequest():

    def get_recent_tweets(
        self
    ):
        response = requests.post(
            f"http://localhost:40810/api/v1/data/twitter/tweets/recent",
            json={"query": '("eth") since:2024-10-16', "count": 200},
            headers={"accept": "application/json", "Content-Type": "application/json"},
            timeout=90,
        )

        if response.ok:
            data = dict(response.json()).get("data", [])

            bt.logging.info("TWEETS===START")
            #bt.logging.success(f"Getting Total: {len(data)}")

            # print("===TRUOC===")
            # print(data)
            # print("===ENDTRUOC===")

            data = self.update_data(data)
            bt.logging.debug(data[:3])

            # print("===SAU===")
            # print(data)
            # print("===ENDSAU===")

            #bt.logging.success(f"Sending Total: {len(data)} tweets to validator")
            bt.logging.info("TWEETS===END")
            return data
        else:
            bt.logging.error(
                f"Twitter recent tweets request failed with status code: {response.status_code}"
            )
            return None

    def update_data(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def calculate_similarity_percentage(response_embedding, source_embedding):
            cosine_similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(response_embedding).unsqueeze(0),
                torch.tensor(source_embedding).unsqueeze(0),
            ).item()
            similarity_percentage = (cosine_similarity + 1) / 2 * 100
            return similarity_percentage
        
        def update_fields(data: Dict[str, Any]) -> Dict[str, Any]:
            newData = {}
            
            newData["ID"] = data.get("ID", "")
            newData["Name"] = data.get("Name", "")
            newData["Username"] = data.get("Username", "")
            newData["Text"] = data.get("Text", "")
            newData["Timestamp"] = data.get("Timestamp", 0)
            newData["Hashtags"] = data.get("Hashtags", None)
    
            newData["ConversationID"] = data.get("ConversationID", "")
            newData["GIFs"] = None
            newData["HTML"] = data.get("Text", "") # fix
            newData["InReplyToStatus"] = data.get("InReplyToStatus", None)
            newData["InReplyToStatusID"] = data.get("InReplyToStatus", None)
            newData["IsQuoted"] = data.get("IsQuoted", False)
            newData["IsPin"] = data.get("IsPin", False)
            newData["IsReply"] = data.get("IsReply", False)
            newData["IsRetweet"] = data.get("IsRetweet", False)
            newData["IsSelfThread"] = data.get("IsSelfThread", False)
            newData["Likes"] = data.get("Likes", 0)
            newData["Mentions"] = None # fix
            newData["PermanentURL"] = "" # fix
            newData["Photos"] = None # fix
            newData["Place"] = None # fix
            newData["QuotedStatus"] = data.get("QuotedStatus", None)
            newData["QuotedStatusID"] = data.get("QuotedStatusID", None)
            newData["Replies"] = data.get("QuotedStatusID", 0)
            newData["Retweets"] = data.get("Retweets", 0)
            newData["RetweetedStatus"] = data.get("RetweetedStatus", None)
            newData["RetweetedStatusID"] = data.get("RetweetedStatusID", None)
            newData["Thread"] = data.get("Thread", None)
            newData["TimeParsed"] = data.get("TimeParsed", "")
            newData["URLs"] = None # fix
            newData["UserID"] = data.get("UserID", "")
            newData["Videos"] = None # fix
            newData["Views"] = data.get("Views", 0)
            newData["SensitiveContent"] = data.get("SensitiveContent", False)
        
            return ProtocolTwitterTweetResponse(
                Tweet=newData,
                Error={"details": "", "error": "", "workerPeerId": ""},
            )
        
        model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        example_tweet = ProtocolTwitterTweetResponse(
            Tweet={
                "ConversationID": "",
                "GIFs": None,
                "Hashtags": None,
                "HTML": "",
                "ID": "",
                "InReplyToStatus": None,
                "InReplyToStatusID": None,
                "IsQuoted": False,
                "IsPin": False,
                "IsReply": False,
                "IsRetweet": False,
                "IsSelfThread": False,
                "Likes": 0,
                "Mentions": None,
                "Name": "",
                "PermanentURL": "",
                "Photos": None,
                "Place": None,
                "QuotedStatus": None,
                "QuotedStatusID": None,
                "Replies": 0,
                "Retweets": 0,
                "RetweetedStatus": None,
                "RetweetedStatusID": None,
                "Text": "",
                "Thread": None,
                "TimeParsed": "",
                "Timestamp": 0,
                "URLs": None,
                "UserID": "",
                "Username": "",
                "Videos": None,
                "Views": 0,
                "SensitiveContent": False,
            },
            Error={"details": "", "error": "", "workerPeerId": ""},
        )
        example_tweet_embedding = model.encode(str(example_tweet))

        def checkValidSimilarity(tweet_data):
            tweet_embedding = model.encode(str(tweet_data))
            similarity = (
                calculate_similarity_percentage(
                    example_tweet_embedding,
                    tweet_embedding,
                )
            )
            if similarity >= 90:  # pretty strict
                return True
            else:
                print(f"similarity: {similarity}")
                print(tweet_data)

            return False

        # 1. Đếm số lượng tweet hợp lệ trong danh sách cũ
        old_tweets_valid = sum(1 for tweet_data in tweets if checkValidSimilarity(tweet_data))

        # 2. Cập nhật tất cả các tweet trong danh sách
        new_tweets = [update_fields(tweet_data["Tweet"]) or tweet_data for tweet_data in tweets]

        # 3. Đếm số lượng tweet hợp lệ trong danh sách mới
        print('NEW:')
        new_tweets_valid = sum(1 for tweet_data in new_tweets if checkValidSimilarity(tweet_data))

        countResponse = 0 if not tweets else len(tweets)
        print(f"old_tweets_valid={old_tweets_valid}/{countResponse} VS new_tweets_valid={new_tweets_valid}/{countResponse}")
        return new_tweets

TwitterTweetsRequest().get_recent_tweets()