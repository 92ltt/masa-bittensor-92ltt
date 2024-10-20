import bittensor as bt
from typing import List, Optional, Any, Dict
from masa.miner.masa_protocol_request import MasaProtocolRequest
from masa.types.twitter import ProtocolTwitterTweetResponse
from sentence_transformers import SentenceTransformer
import torch

class RecentTweetsSynapse(bt.Synapse):
    query: str
    count: int
    response: Optional[Any] = None

    def deserialize(self) -> Optional[Any]:
        return self.response


def forward_recent_tweets(synapse: RecentTweetsSynapse) -> RecentTweetsSynapse:
    synapse.response = TwitterTweetsRequest().get_recent_tweets(synapse)
    return synapse


class TwitterTweetsRequest(MasaProtocolRequest):
    def __init__(self):
        super().__init__()

    def get_recent_tweets(
        self, synapse: RecentTweetsSynapse
    ) -> Optional[List[ProtocolTwitterTweetResponse]]:
        bt.logging.info(f"Getting {synapse.count} recent tweets for: {synapse.query}")
        response = self.post(
            "/data/twitter/tweets/recent",
            body={"query": synapse.query, "count": synapse.count},
        )

        if response.ok:
            data = self.format(response)
            bt.logging.info("TWEETS===START")

            countResponse = 0 if not data else len(data)
            bt.logging.success(f"Getting Total: {countResponse}")

            data = self.update_data(data)

            countResponse = 0 if not data else len(data)
            bt.logging.success(f"Sending Total: {countResponse} tweets to validator")

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
            if similarity >= 60:  # pretty strict
                return True

            return False

        # 1. Đếm số lượng tweet hợp lệ trong danh sách cũ
        old_tweets_valid = sum(1 for tweet_data in tweets if checkValidSimilarity(tweet_data))

        # 2. Cập nhật tất cả các tweet trong danh sách
        new_tweets = [update_fields(tweet_data["Tweet"]) or tweet_data for tweet_data in tweets]

        # 3. Đếm số lượng tweet hợp lệ trong danh sách mới
        new_tweets_valid = sum(1 for tweet_data in new_tweets if checkValidSimilarity(tweet_data))

        countResponse = 0 if not tweets else len(tweets)
        #print(f"old_tweets_valid={old_tweets_valid}/{countResponse} VS new_tweets_valid={new_tweets_valid}/{countResponse}")
        bt.logging.info(f"old_tweets_valid={old_tweets_valid}/{countResponse} VS new_tweets_valid={new_tweets_valid}/{countResponse}")
        return new_tweets