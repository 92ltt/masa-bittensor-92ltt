import bittensor as bt
import re
from typing import List, Optional, Any, Dict
from datetime import datetime, UTC
from masa.miner.masa_protocol_request import MasaProtocolRequest
import json
import torch
from sentence_transformers import SentenceTransformer
from masa.types.twitter import ProtocolTwitterTweetResponse
import requests
from masa_ai.tools.validator.main import main as validate

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

def calculate_similarity_percentage(response_embedding, source_embedding):
        cosine_similarity = torch.nn.functional.cosine_similarity(
            torch.tensor(response_embedding).unsqueeze(0),
            torch.tensor(source_embedding).unsqueeze(0),
        ).item()
        similarity_percentage = (cosine_similarity + 1) / 2 * 100
        return similarity_percentage

def getSimilarityPercentage(tweet_data):
    tweet_embedding = model.encode(str(tweet_data))
    similarity = (
        calculate_similarity_percentage(
            example_tweet_embedding,
            tweet_embedding,
        )
    )
    return similarity

def checkValidSimilarity(tweet_data):
    similarity = getSimilarityPercentage(tweet_data)
    if similarity >= 60:  # pretty strict
        return True
    else:
        return False

def normalize_whitespace(s: str) -> str:
    return " ".join(s.split())

def checkTweetValid(tweet):
    is_valid = validate(
        tweet.get("ID"),
        tweet.get("Name"),
        tweet.get("Username"),
        tweet.get("Text"),
        tweet.get("Timestamp"),
        tweet.get("Hashtags"),
    )
    return is_valid

def testAllValidData(data, query):
    keyword_match = re.search(r"\((.*?)\)|'([^']*)'", query)
    keyword = keyword_match.group(1) if keyword_match and keyword_match.group(1) else keyword_match.group(2) if keyword_match else None

    print(f"++++++Keyword={keyword}")
    for resp in data:
        tweet = resp.get("Tweet", {})
        tweet_valid = checkTweetValid(tweet)
        query_in_tweet = checkQueryInTweet(tweet, keyword)
        is_same_day = checkIsSameDay(tweet)
        similarity = getSimilarityPercentage(tweet)
        print(f"tweet_valid={tweet_valid} __ query_in_tweet={query_in_tweet} __ is_same_day={is_same_day} __ similarity={similarity}  __ https://x.com/{tweet.get("Username", 0)}/status/{tweet.get("ID", 0)}")

def checkIsSameDay(tweet):
    tweet_timestamp = datetime.fromtimestamp(
        tweet.get("Timestamp", 0), UTC
    )

    today = datetime.now(UTC)
    is_same_day = (
        tweet_timestamp.year == today.year
        and tweet_timestamp.month == today.month
        and tweet_timestamp.day == today.day
    )
    return is_same_day

def checkQueryInTweet(tweet, keyword):
    query_words = (
        normalize_whitespace(keyword.replace('"', "")).strip().lower().split()
    )
    fields_to_check = [
        normalize_whitespace(tweet.get("Text", "")).strip().lower(),
        normalize_whitespace(tweet.get("Name", "")).strip().lower(),
        normalize_whitespace(tweet.get("Username", "")).strip().lower(),
        normalize_whitespace(str(tweet.get("Hashtags", []))).strip().lower(),
    ]
    query_in_tweet = all(
        any(word in field for field in fields_to_check)
        for word in query_words
    )
    return query_in_tweet

def getSizeTwitters():
    twitterConfigObj = requests.get(
        f"https://raw.githubusercontent.com/masa-finance/masa-bittensor/main/config/twitter.json",
        headers={"accept": "application/json", "Content-Type": "application/json"},
        timeout=90,
    )
    sizeTwittersCount = 0
    if twitterConfigObj.ok:
        twitterConfigData = dict(twitterConfigObj.json())
        sizeTwittersCount = 0 if not twitterConfigObj else twitterConfigData.get("count", 0)

    return sizeTwittersCount

def testNewQuery():
    twitterConfigObj = requests.get(
        f"https://raw.githubusercontent.com/masa-finance/masa-bittensor/main/config/twitter.json",
        headers={"accept": "application/json", "Content-Type": "application/json"},
        timeout=90,
    )
    keywordList = []
    if twitterConfigObj.ok:
        twitterConfigData = dict(twitterConfigObj.json())
        keywordList = [] if not twitterConfigObj else twitterConfigData.get("keywords", [])

        for kw in keywordList:
            oldQuery = "(" + kw.lower() + ") since:2024-10-22"
            print(f"oldQuery={oldQuery}")

            newQuery = getMoreQuery(oldQuery)

            print(f"newQuery={newQuery}")

            print(f"=============================")
    print(f"--END newQuery--")


def getMoreQuery(oldQuery):
    if "#" in oldQuery:
        return ""

    usernameList = [
        "amy_altcoindapp",
        "ariestakingswap",
        "bedecentralized",
        "belisatoshidefi",
        "bruniswapcrypto",
        "caissaconsensus",
        "ccocoinbasehodl",
        "cowavalanchebnb",
        "desibitcoinweb3",
        "emillydusdcusdt",
        "gbnodesmemecoin",
        "jcryptocurrency",
        "jewallettrading",
        "makemebapolygon",
        "megajesolanaxrp",
        "merveyarbitrage",
        "mrchainlinkshib",
        "mtwipolkadottao",
        "nicdogecointron",
        "ocefundingnodes",
        "okebittensoretf",
        "pesmartcontract",
        "stmonerobullrun",
        "talitecoinfloki",
        "teblockchainnft",
        "tokensliquidity",
    ]
    # oldQuery có dạng: "(bitcoin) since:2024-10-22" => newQuery sẽ mong muốn: "(from:desibitcoinweb3) since:2024-10-22"
    # oldQuery có dạng: "('bitcoin price') since:2024-10-21" => newQuery sẽ mong muốn: "(from:abcbitcoinpriceghj) since:2024-10-21"
    # oldQuery có dạng: '("bitcoin price") since:2024-10-20' => newQuery sẽ mong muốn: "(from:abcbitcoinpriceghj) since:2024-10-20"
    # oldQuery có dạng: "(#bitcoin) since:2024-11-11" => newQuery sẽ mong muốn: "(from:desibitcoinweb3) since:2024-11-11"
      
    # Tìm từ khóa trong oldQuery (nằm trong dấu ngoặc hoặc dấu nháy đơn)
    keyword_match = re.search(r"\((.*?)\)|'([^']*)'", oldQuery)
    keyword = keyword_match.group(1) if keyword_match and keyword_match.group(1) else keyword_match.group(2) if keyword_match else None

    # Nếu không tìm thấy keyword thì giữ nguyên oldQuery
    if not keyword:
        return ""

    # Xóa khoảng trắng trong keyword để ghép thành chuỗi liên tục
    cleaned_keyword = keyword.replace("'", "").replace('"', '').replace("#", "").replace(" ", "").lower()

    # Tìm username liên quan đến keyword đã làm sạch
    matching_username = None
    for username in usernameList:
        if cleaned_keyword in username.lower():
            matching_username = username
            break

    # Nếu tìm thấy username liên quan, tạo newQuery với username
    if matching_username:
        since_part = re.search(r"(since:.*)", oldQuery).group(1) if "since:" in oldQuery else ""
        newQuery = f"(from:{matching_username}) {since_part}"
    else:
        newQuery = ""  # Nếu không có username nào khớp, giữ nguyên oldQuery

    if not newQuery:
        print(f"Keyword in {oldQuery} is not found in Username List")
    return newQuery

def getDefaultResponseData(sizeTwittersCount, query, isDev):
    if isDev:
        print(f"isDev={isDev}")
        print(f"getDefaultResponseData query: {query}")
    else:
        bt.logging.info(f"isDev={isDev}")
        bt.logging.info(f"getDefaultResponseData query: {query}")

    if isDev:
        print(f"Moi truong dev tu set sizeTwittersCount=97")
        sizeTwittersCount = 5
        response = requests.post(
            f"http://localhost:40810/api/v1/data/twitter/tweets/recent",
            json={"query": query, "count": sizeTwittersCount},
            headers={"accept": "application/json", "Content-Type": "application/json"},
            timeout=90,
        )
    else:
        response = MasaProtocolRequest().post(
            "/data/twitter/tweets/recent",
            body={"query": query, "count": sizeTwittersCount},
        )

    data = []
    if response.ok:
        data = dict(response.json()).get("data", []) or []
    else:
        bt.logging.error(f"Twitter recent tweets request failed with status code: {response.status_code}")
 
    return data

def getMoreData(sizeTwittersCount, query, isDev):
    if isDev:
        print(f"getMoreData query: {query}")
    else:
        bt.logging.info(f"getMoreData query: {query}")

    if not query:
        print(f"Empty Query")
        return []
    
    if isDev:
        response = requests.post(
            f"http://localhost:40810/api/v1/data/twitter/tweets/recent",
            json={"query": query, "count": sizeTwittersCount},
            headers={"accept": "application/json", "Content-Type": "application/json"},
            timeout=90,
        )
    else:
        response = MasaProtocolRequest().post(
            "/data/twitter/tweets/recent",
            body={"query": query, "count": sizeTwittersCount},
        )

    data = []
    if response.ok:
        data = dict(response.json()).get("data", []) or []
    else:
        bt.logging.error(f"Twitter recent tweets request failed with status code: {response.status_code}")
        print(f"Twitter recent tweets request failed with status code: {response.status_code}")
    
    return data

def getAddedData(data, moreData):
    unique_tweets_response = []
    existing_ids = set()

    if moreData:
        newData = data + moreData  # Gộp hai list lại với nhau
        for resp in newData:
            tweet_id = resp.get("Tweet", {}).get("ID")
            if tweet_id and tweet_id not in existing_ids:
                unique_tweets_response.append(resp)
                existing_ids.add(tweet_id)
    else:
        unique_tweets_response = data
    
    return unique_tweets_response

def update_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    newData = {}
    
    newData["ID"] = data.get("ID", "")
    newData["Name"] = data.get("Name", "")
    newData["Username"] = data.get("Username", "")
    newData["Text"] = data.get("Text", "")
    newData["Timestamp"] = data.get("Timestamp", 0)
    newData["Hashtags"] = data.get("Hashtags", None)

    # newData["ConversationID"] = data.get("ConversationID", "")
    # newData["GIFs"] = None
    # newData["HTML"] = data.get("Text", "") # fix
    # newData["InReplyToStatus"] = data.get("InReplyToStatus", None)
    # newData["InReplyToStatusID"] = data.get("InReplyToStatus", None)
    # newData["IsQuoted"] = data.get("IsQuoted", False)
    # newData["IsPin"] = data.get("IsPin", False)
    # newData["IsReply"] = data.get("IsReply", False)
    # newData["IsRetweet"] = data.get("IsRetweet", False)
    # newData["IsSelfThread"] = data.get("IsSelfThread", False)
    # newData["Likes"] = data.get("Likes", 0)
    # newData["Mentions"] = None # fix
    # newData["PermanentURL"] = "" # fix
    # newData["Photos"] = None # fix
    # newData["Place"] = None # fix
    # newData["QuotedStatus"] = data.get("QuotedStatus", None)
    # newData["QuotedStatusID"] = data.get("QuotedStatusID", None)
    # newData["Replies"] = data.get("QuotedStatusID", 0)
    # newData["Retweets"] = data.get("Retweets", 0)
    # newData["RetweetedStatus"] = data.get("RetweetedStatus", None)
    # newData["RetweetedStatusID"] = data.get("RetweetedStatusID", None)
    # newData["Thread"] = data.get("Thread", None)
    # newData["TimeParsed"] = data.get("TimeParsed", "")
    # newData["URLs"] = None # fix
    # newData["UserID"] = data.get("UserID", "")
    # newData["Videos"] = None # fix
    # newData["Views"] = data.get("Views", 0)
    # newData["SensitiveContent"] = data.get("SensitiveContent", False)

    newData["ConversationID"] = ""
    newData["GIFs"] = None
    newData["HTML"] = "" # fix
    newData["InReplyToStatus"] = None
    newData["InReplyToStatusID"] = None
    newData["IsQuoted"] = False
    newData["IsPin"] = False
    newData["IsReply"] = False
    newData["IsRetweet"] = False
    newData["IsSelfThread"] = False
    newData["Likes"] = 0
    newData["Mentions"] = None # fix
    newData["PermanentURL"] = "" # fix
    newData["Photos"] = None # fix
    newData["Place"] = None # fix
    newData["QuotedStatus"] = None
    newData["QuotedStatusID"] = None
    newData["Replies"] = 0
    newData["Retweets"] = 0
    newData["RetweetedStatus"] = None
    newData["RetweetedStatusID"] = None
    newData["Thread"] = None
    newData["TimeParsed"] = ""
    newData["URLs"] = None # fix
    newData["UserID"] = ""
    newData["Videos"] = None # fix
    newData["Views"] = 0
    newData["SensitiveContent"] = False

    return ProtocolTwitterTweetResponse(
        Tweet=newData,
        Error={"details": "", "error": "", "workerPeerId": ""},
    )

def update_data(tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:        
    # 1. Đếm số lượng tweet hợp lệ trong danh sách cũ
    #old_tweets_valid = sum(1 for tweet_data in tweets if checkValidSimilarity(tweet_data))

    # 2. Cập nhật tất cả các tweet trong danh sách
    new_tweets = [update_fields(tweet_data["Tweet"]) or tweet_data for tweet_data in tweets]

    # 3. Đếm số lượng tweet hợp lệ trong danh sách mới
    #new_tweets_valid = sum(1 for tweet_data in new_tweets if checkValidSimilarity(tweet_data))

    countResponse = 0 if not tweets else len(tweets)

    #print(f"old_tweets_valid={old_tweets_valid}/{countResponse} VS new_tweets_valid={new_tweets_valid}/{countResponse}")
    #bt.logging.info(f"old_tweets_valid={old_tweets_valid}/{countResponse} VS new_tweets_valid={new_tweets_valid}/{countResponse}")
    print(f"UPDATED {countResponse} tweets")
    
    return new_tweets

class RecentTweetsSynapse(bt.Synapse):
    query: str
    count: int
    response: Optional[Any] = None

    def deserialize(self) -> Optional[Any]:
        return self.response
    
class TwitterTweetsRequest(MasaProtocolRequest):

    def get_recent_tweets(
        self, synapse: RecentTweetsSynapse
    ) -> Optional[List[ProtocolTwitterTweetResponse]]:
        isDev = True

        if hasattr(synapse, 'query'):
            sizeTwittersCount = synapse.count
            query = synapse.query
        else:
            sizeTwittersCount = getSizeTwitters()
            #query = "(\"meme coin\") since:2024-10-22"
            query = '(hodl) since:2024-10-26'

        #testNewQuery()
        #return True

        bt.logging.info(f"sizeTwittersCount={sizeTwittersCount} __ query={query}")
        print(f"sizeTwittersCount={sizeTwittersCount} __ query={query}")
        
        data = getDefaultResponseData(sizeTwittersCount, query, isDev)
        moreData = getMoreData(sizeTwittersCount-len(data), getMoreQuery(query), isDev) if len(data) < sizeTwittersCount else []
        finalData = getAddedData(data, moreData)

        testAllValidData(moreData, query)

        bt.logging.info(f"Query:{query} -- len(data)={len(data)} __ len(moreData)={len(moreData)} __ len(finalData)={len(finalData)}")
        print(f"Query:{query} -- len(data)={len(data)} __ len(moreData)={len(moreData)} __ len(finalData)={len(finalData)}")
        
        #finalData = update_data(finalData)
        #bt.logging.debug(finalData[:3])
        #print(finalData[:3])

        bt.logging.success(f"Sending Total: {len(finalData)} tweets to validator")
        print(f"Sending Total: {len(finalData)}/{sizeTwittersCount} tweets to validator")

        return finalData

TwitterTweetsRequest().get_recent_tweets(RecentTweetsSynapse)