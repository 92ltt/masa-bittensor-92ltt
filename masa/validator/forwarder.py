# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Masa

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import bittensor as bt
from typing import Any
from datetime import datetime, UTC
import aiohttp
import random
import json

from masa.miner.twitter.tweets import RecentTweetsSynapse
from masa.miner.twitter.profile import TwitterProfileSynapse
from masa.miner.twitter.followers import TwitterFollowersSynapse

from masa.base.healthcheck import PingAxonSynapse, get_external_ip
from masa.utils.uids import get_random_miner_uids

from masa_ai.tools.validator.main import main as validate

TIMEOUT = 8


class Forwarder:
    def __init__(self, validator):
        self.validator = validator

    async def forward_request(
        self, request: Any, sample_size: int, timeout: int = TIMEOUT
    ):
        miner_uids = await get_random_miner_uids(self.validator, k=sample_size)
        dendrite = bt.dendrite(wallet=self.validator.wallet)
        responses = await dendrite(
            [self.validator.metagraph.axons[uid] for uid in miner_uids],
            request,
            deserialize=True,
            timeout=timeout,
        )

        formatted_responses = [
            {"uid": int(uid), "response": response}
            for uid, response in zip(miner_uids, responses)
        ]
        return formatted_responses, miner_uids

    async def get_twitter_profile(self, username: str = "getmasafi"):
        request = TwitterProfileSynapse(username=username)
        formatted_responses, _ = await self.forward_request(
            request=request, sample_size=self.validator.config.neuron.sample_size
        )
        return formatted_responses

    async def get_twitter_followers(self, username: str = "getmasafi", count: int = 10):
        request = TwitterFollowersSynapse(username=username, count=count)
        formatted_responses, _ = await self.forward_request(
            request=request, sample_size=self.validator.config.neuron.sample_size
        )
        return formatted_responses

    async def get_recent_tweets(
        self,
        query: str = f"(Bitcoin) since:{datetime.now().strftime('%Y-%m-%d')}",
        count: int = 3,
    ):
        request = RecentTweetsSynapse(query=query, count=count)
        formatted_responses, _ = await self.forward_request(
            request=request, sample_size=self.validator.config.neuron.sample_size
        )
        return formatted_responses

    async def get_discord_profile(self, user_id: str = "449222160687300608"):
        return ["Not yet implemented"]

    async def get_discord_channel_messages(self, channel_id: str):
        return ["Not yet implemented"]

    async def get_discord_guild_channels(self, guild_id: str):
        return ["Not yet implemented"]

    async def get_discord_user_guilds(self):
        return ["Not yet implemented"]

    async def get_discord_all_guilds(self):
        return ["Not yet implemented"]

    async def ping_axons(self):
        request = PingAxonSynapse(
            sent_from=get_external_ip(), is_active=False, version=0
        )
        sample_size = self.validator.config.neuron.sample_size_ping
        dendrite = bt.dendrite(wallet=self.validator.wallet)
        all_responses = []
        for i in range(0, len(self.validator.metagraph.axons), sample_size):
            batch = self.validator.metagraph.axons[i : i + sample_size]
            batch_responses = await dendrite(
                batch,
                request,
                deserialize=False,
                timeout=TIMEOUT,
            )
            all_responses.extend(batch_responses)

        self.validator.versions = [response.version for response in all_responses]
        bt.logging.info(f"Miner Versions: {self.validator.versions}")
        self.validator.last_tempo_block = self.validator.subtensor.block

        return [
            {
                "status_code": response.dendrite.status_code,
                "status_message": response.dendrite.status_message,
                "version": response.version,
                "uid": all_responses.index(response),
            }
            for response in all_responses
        ]

    async def fetch_twitter_config(self):
        async with aiohttp.ClientSession() as session:
            url = self.validator.config.neuron.twitter_config_url
            async with session.get(url) as response:
                if response.status == 200:
                    configRaw = await response.text()
                    config = json.loads(configRaw)
                    bt.logging.info(f"@@Twitter config url!: {url}")
                    bt.logging.info(f"Twitter config fetched!: {config}")
                    self.validator.keywords = config["keywords"]
                    self.validator.count = int(config["count"])
                else:
                    bt.logging.error(
                        f"Failed to fetch config from GitHub: {response.status}"
                    )
                    # note, defaults
                    self.validator.keywords = ["crypto", "bitcoin", "masa"]
                    self.validator.count = 3

    async def get_miners_volumes(self):
        if len(self.validator.versions) == 0:
            bt.logging.info("Pinging axons to get miner versions...")
            return await self.ping_axons()
        if len(self.validator.keywords) == 0 or self.check_tempo():
            await self.fetch_twitter_config()

        random_keyword = random.choice(self.validator.keywords)
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        query = f"({random_keyword.strip()}) since:{today}"
        bt.logging.debug(f"Checked-Query {query}:")
        request = RecentTweetsSynapse(query=query, count=self.validator.count)

        responses, miner_uids = await self.forward_request(
            request, sample_size=self.validator.config.neuron.sample_size_volume
        )

        all_valid_tweets = []
        for response, uid in zip(responses, miner_uids):
            bt.logging.debug(f"START -- UID: {uid}")
            valid_tweets = []
            all_responses = dict(response).get("response", [])
            if not all_responses:
                continue

            bt.logging.debug(f"H-Query all_responses:")
            bt.logging.debug(len(all_responses))

            unique_tweets_response = []
            existing_ids = set()
            for resp in all_responses:
                tweet_id = resp.get("Tweet", {}).get("ID")
                if tweet_id and tweet_id not in existing_ids:
                    unique_tweets_response.append(resp)
                    existing_ids.add(tweet_id)

            if unique_tweets_response is not None:
                bt.logging.debug(f"H-Query unique_tweets_response:")
                bt.logging.debug(len(unique_tweets_response))

                # note, first spot check this payload, ensuring a random tweet is valid
                random_tweet = dict(random.choice(unique_tweets_response)).get(
                    "Tweet", {}
                )

                is_valid = validate(
                    random_tweet.get("ID"),
                    random_tweet.get("Name"),
                    random_tweet.get("Username"),
                    random_tweet.get("Text"),
                    random_tweet.get("Timestamp"),
                    random_tweet.get("Hashtags"),
                )
                bt.logging.debug(f"H-Query is_valid: {is_valid} ===> https://x.com/{random_tweet.get("Username")}/status/{random_tweet.get("ID")} - {query}")

                query_words = (
                    self.normalize_whitespace(random_keyword.replace('"', ""))
                    .strip()
                    .lower()
                    .split()
                )

                bt.logging.debug(f"H-Query query_words: {query_words}")

                fields_to_check = [
                    self.normalize_whitespace(random_tweet.get("Text", ""))
                    .strip()
                    .lower(),
                    self.normalize_whitespace(random_tweet.get("Name", ""))
                    .strip()
                    .lower(),
                    self.normalize_whitespace(random_tweet.get("Username", ""))
                    .strip()
                    .lower(),
                    self.normalize_whitespace(str(random_tweet.get("Hashtags", [])))
                    .strip()
                    .lower(),
                ]

                query_in_tweet = all(
                    any(word in field for field in fields_to_check)
                    for word in query_words
                )

                bt.logging.debug(f"H-Query query_in_tweet: {query_in_tweet}")
                if not query_in_tweet:
                    bt.logging.warning(
                        f"Query: {random_keyword} is not in the tweet: {fields_to_check}"
                    )

                tweet_timestamp = datetime.fromtimestamp(
                    random_tweet.get("Timestamp", 0), UTC
                )

                today = datetime.now(UTC)
                is_same_day = (
                    tweet_timestamp.year == today.year
                    and tweet_timestamp.month == today.month
                    and tweet_timestamp.day == today.day
                )

                if not is_same_day:
                    bt.logging.warning(
                        f"Tweet timestamp {tweet_timestamp} is not from today {today}"
                    )

                # note, they passed the spot check!
                if is_valid and query_in_tweet and is_same_day:
                    bt.logging.success(
                        f"Miner {uid} passed the spot check with query: {random_keyword}"
                    )
                    for tweet in unique_tweets_response[
                        : self.validator.count
                    ]:  # note, limits to the count requested
                        if tweet:
                            tweet_embedding = self.validator.model.encode(str(tweet))
                            similarity = (
                                self.validator.scorer.calculate_similarity_percentage(
                                    self.validator.example_tweet_embedding,
                                    tweet_embedding,
                                )
                            )
                            if similarity >= 60:  # pretty strict
                                valid_tweets.append(tweet)
                            else:
                                bt.logging.debug(f"NotPassData {uid} : similarity {similarity} < 60")
                                bt.logging.debug(tweet)
                else:
                    bt.logging.warning(f"Miner {uid} failed the spot check! is_valid={is_valid} query_in_tweet={query_in_tweet} today={today}")
                    bt.logging.debug(tweet)

            bt.logging.debug(f"H-Result: uid {uid} len(all_responses)={len(all_responses)} VS len(valid_tweets)={len(valid_tweets)}")

            self.validator.scorer.add_volume(int(uid), len(valid_tweets))
            bt.logging.info(f"Miner {uid} produced {len(valid_tweets)} valid tweets")
            all_valid_tweets.extend(valid_tweets)

        bt.logging.debug(f"H-Query self.validator.indexed_tweets:")
        bt.logging.debug(len(self.validator.indexed_tweets))
        query_exists = False
        for indexed_tweet in self.validator.indexed_tweets:
            if indexed_tweet["query"] == query:
                bt.logging.debug(f"@@ - came here 1")
                existing_tweet_ids = {
                    tweet["Tweet"]["ID"] for tweet in indexed_tweet["tweets"]
                }
                unique_valid_tweets = [
                    tweet
                    for tweet in all_valid_tweets
                    if tweet["Tweet"]["ID"] not in existing_tweet_ids
                ]
                indexed_tweet["tweets"].extend(unique_valid_tweets)
                query_exists = True
                break
        bt.logging.debug(f"@@ - came here 2")
        if not query_exists:
            bt.logging.debug(f"@@ - came here 3")
            payload = {
                "query": query,
                "tweets": [
                    tweet
                    for tweet in {
                        tweet["Tweet"]["ID"]: tweet
                        for tweet in all_valid_tweets  # note, only unique tweets
                    }.values()
                ],
            }
            self.validator.indexed_tweets.append(payload)

        bt.logging.debug(f"@@ - came here 4")

        # note, set the last volume block to the current block
        self.validator.last_volume_block = self.validator.subtensor.block

    def check_tempo(self) -> bool:
        if self.validator.last_tempo_block == 0:
            self.validator.last_tempo_block = self.validator.subtensor.block
            return True

        tempo = self.validator.tempo
        blocks_since_last_check = (
            self.validator.subtensor.block - self.validator.last_tempo_block
        )
        if blocks_since_last_check >= tempo:
            self.validator.last_tempo_block = self.validator.subtensor.block
            return True
        else:
            return False

    def normalize_whitespace(self, s: str) -> str:
        return " ".join(s.split())
