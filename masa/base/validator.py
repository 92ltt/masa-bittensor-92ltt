# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Masa

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import copy
import torch
import asyncio
import argparse
import threading
import bittensor as bt

from typing import List

from masa.base.neuron import BaseNeuron
from masa.utils.config import add_validator_args
from masa.mock import MockDendrite

from masa.validator.scorer import Scorer
from masa.validator.forwarder import Forwarder
from sentence_transformers import SentenceTransformer

from masa.types.twitter import ProtocolTwitterTweetResponse
from masa.utils.weights import process_weights_for_netuid


class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    neuron_type: str = "ValidatorNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        self.forwarder = Forwarder(self)
        self.scorer = Scorer(self)
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )  # Load a pre-trained model for embeddings
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
        self.example_tweet_embedding = self.model.encode(str(example_tweet))

        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        self.tempo = self.subtensor.get_subnet_hyperparameters(self.config.netuid).tempo

        self.last_sync_block = 0
        self.last_tempo_block = 0
        self.last_volume_block = 0
        self.last_scoring_block = 0

        self.keywords = []  # note, for volume queries
        self.count = 0  # note, for volume queries

        if self.config.subtensor._mock:
            self.dendrite = MockDendrite(wallet=self.wallet)
        else:
            self.dendrite = bt.dendrite(wallet=self.wallet)

        self.scores = torch.zeros(
            self.metagraph.n, dtype=torch.float32, device=self.device
        )
        # Init sync with the network. Updates the metagraph.
        self.sync()
        self.load_state()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.sync_thread: threading.Thread = None
        self.miner_ping_thread: threading.Thread = None
        self.miner_volume_thread: threading.Thread = None
        self.miner_scoring_thread: threading.Thread = None
        self.auto_update_thread: threading.Thread = None
        self.lock = asyncio.Lock()

        self.run_in_background_thread()

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(
                wallet=self.wallet, config=self.config, port=self.config.axon.port
            )

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                bt.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(f"Failed to create Axon initialize with exception: {e}")
            pass

    async def run_sync(self):
        while not self.should_exit:
            try:
                blocks_since_last_check = self.block - self.last_sync_block
                if blocks_since_last_check >= 6:
                    self.sync()
                    self.step += 1
                    self.last_sync_block = self.block
            except Exception as e:
                bt.logging.error(f"Error running sync: {e}")
            # TODO is there a better way to go about this?
            await asyncio.sleep(3 * 12)

    async def run_miner_ping(self):
        while not self.should_exit:
            try:
                if self.forwarder.check_tempo():
                    await self.forwarder.ping_axons()
            except Exception as e:
                bt.logging.error(f"Error running miner ping: {e}")
            # TODO is there a better way to go about this?
            await asyncio.sleep(self.tempo / 2 * 12)

    async def run_miner_volume(self):
        while not self.should_exit:
            try:
                blocks_since_last_check = self.block - self.last_volume_block
                if blocks_since_last_check >= self.tempo / 100:
                    await self.forwarder.get_miners_volumes()
            except Exception as e:
                bt.logging.error(f"Error running miner volume: {e}")
            # TODO is there a better way to go about this?
            await asyncio.sleep(self.tempo / 200 * 12)

    async def run_miner_scoring(self):
        while not self.should_exit:
            try:
                blocks_since_last_check = self.block - self.last_scoring_block
                if blocks_since_last_check >= self.tempo / 50:
                    await self.scorer.score_miner_volumes()
            except Exception as e:
                bt.logging.error(f"Error running miner scoring: {e}")
            # TODO is there a better way to go about this?
            await asyncio.sleep(self.tempo / 100 * 12)

    # note, runs every tempo
    async def run_auto_update(self):
        while not self.should_exit:
            try:
                if self.config.neuron.auto_update:
                    self.auto_update()
            except Exception as e:
                bt.logging.error(f"Error running auto update: {e}")
            await asyncio.sleep(self.tempo * 12)  # note, 12 seconds per block

    def run_sync_in_loop(self):
        asyncio.run(self.run_sync())

    def run_miner_ping_in_loop(self):
        asyncio.run(self.run_miner_ping())

    def run_miner_volume_in_loop(self):
        asyncio.run(self.run_miner_volume())

    def run_miner_scoring_in_loop(self):
        asyncio.run(self.run_miner_scoring())

    def run_auto_update_in_loop(self):
        asyncio.run(self.run_auto_update())

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.sync_thread = threading.Thread(
                target=self.run_sync_in_loop, daemon=True
            )
            self.miner_ping_thread = threading.Thread(
                target=self.run_miner_ping_in_loop, daemon=True
            )
            self.miner_volume_thread = threading.Thread(
                target=self.run_miner_volume_in_loop, daemon=True
            )
            self.miner_scoring_thread = threading.Thread(
                target=self.run_miner_scoring_in_loop, daemon=True
            )
            self.auto_update_thread = threading.Thread(
                target=self.run_auto_update_in_loop, daemon=True
            )
            self.sync_thread.start()  # for setting weights, syncing metagraph,, etc
            self.miner_ping_thread.start()  # for versioning and getting keywords
            self.miner_volume_thread.start()  # for testing miner volumes
            self.miner_scoring_thread.start()  # for scoring miner volumes
            self.auto_update_thread.start()  # for auto updating the neuron
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.sync_thread.join(5)
            self.miner_ping_thread.join(5)
            self.miner_volume_thread.join(5)
            self.miner_scoring_thread.join(5)
            self.auto_update_thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.sync_thread.join(5)
            self.miner_ping_thread.join(5)
            self.miner_volume_thread.join(5)
            self.miner_scoring_thread.join(5)
            self.auto_update_thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """
        # Check if self.scores contains any NaN values and log a warning if it does.
        if torch.isnan(self.scores).any():
            bt.logging.warning(
                "Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Calculate the average reward for each uid across non-zero values.
        raw_weights = torch.nn.functional.normalize(self.scores, p=1, dim=0)
        cpu_raw_weights = raw_weights.to("cpu").numpy()

        bt.logging.info(f"@@subtensor.set_weights |uids: {self.metagraph.uids} |weights: {cpu_raw_weights} |self.config.netuid: {self.config.netuid}  |self.subtensor: {self.subtensor} |self.metagraph: {self.metagraph} ")

#  ****** ^_^ ******* self.subtensor.set_weights
# uids: [0 1 2 3 4 5 6 7]
# weights: [0.11490087 0. 0. 0. 0. 0.5802058636 0.30489326 0. ]
# self.config.netuid: 2 
# self.subtensor: subtensor(local, ws://127.0.0.1:9946)
# self.metagraph: metagraph(netuid:2, n:8, block:107935, network:local) 

        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights.to("cpu").numpy(),
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )

        bt.logging.info("@@process_weights_for_netuid DONE $$$")

        (
            uint_uids,
            uint_weights,
        ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )

        bt.logging.info(f"@@Setting weights: {uint_weights} for uids: {uint_uids}")

        # Set the weights on chain via our subtensor connection.
        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=self.spec_version,
        )
        
        bt.logging.info(f"@@self.subtensor.set_weights |wallet: {self.wallet} |netuid: {self.config.netuid} |uids: {uint_uids} |weights: {uint_weights} |version_key: {self.spec_version}")

        if result is True:
            bt.logging.success("set_weights on chain successfully!")
        else:
            bt.logging.error("set_weights failed", msg)

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = torch.zeros((self.metagraph.n)).to(self.device)
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def update_scores(self, rewards: torch.FloatTensor, uids: List[int]):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""
        
        #rewards=tensor([0.8413, 0.1587], device='cuda:0'), uids=[2, 0]

        # Check if rewards contains NaN values.
        if torch.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = torch.nan_to_num(rewards, 0)

        # Check if `uids` is already a tensor and clone it to avoid the warning.
        if isinstance(uids, torch.Tensor):
            uids_tensor = uids.clone().detach()
        else:
            uids_tensor = torch.tensor(uids).to(self.device)

        # Ensure that the uids_tensor and rewards have the same length
        if len(uids_tensor) != len(rewards):
            raise ValueError("The length of uids_tensor and rewards must be the same.")

        # Ensure self.scores has the required length to accommodate all uids in uids_tensor
        max_uid = uids_tensor.max().item()
        if max_uid >= self.scores.size(0):
            new_size = max_uid + 1
            new_scores = torch.zeros(new_size).to(self.device)
            new_scores[: self.scores.size(0)] = self.scores
            self.scores = new_scores

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_rewards: torch.FloatTensor = self.scores.scatter(
            0, uids_tensor, rewards
        ).to(self.device)

        #Scattered rewards: scattered_rewards=tensor([0.1587, 0.0000, 0.8413, 0.2357, 0.2398, 0.2396, 0.2440, 0.0000],
                                                    
        bt.logging.info(f"Scattered rewards: scattered_rewards={scattered_rewards} rewards={rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]

        alpha: float = self.config.neuron.moving_average_alpha
        self.scores: torch.FloatTensor = alpha * scattered_rewards + (
            1 - alpha
        ) * self.scores.to(self.device)

        bt.logging.info(f"Updated moving averages: {self.scores}")

        self.save_state()

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
        torch.save(
            {
                "step": self.step,
                "scores": self.scores,
                "hotkeys": self.hotkeys,
                "volumes": self.volumes,
                "versions": self.versions,
                "indexed_tweets": self.indexed_tweets,
            },
            self.config.neuron.full_path + "/state.pt",
        )

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")

        # Load the state of the validator from file.
        state_path = self.config.neuron.full_path + "/state.pt"
        if os.path.isfile(state_path):
            state = torch.load(state_path)
            self.step = dict(state).get("step", 0)
            self.scores = dict(state).get("scores", [])
            self.hotkeys = dict(state).get("hotkeys", [])
            self.volumes = dict(state).get("volumes", [])
            self.versions = dict(state).get("versions", [])
            self.indexed_tweets = dict(state).get("indexed_tweets", [])
        else:
            self.step = 0
            self.scores = torch.zeros(self.metagraph.n)
            self.hotkeys = []
            self.volumes = []
            self.versions = []
            self.indexed_tweets = []
            bt.logging.warning(
                f"State file not found at {state_path}. Skipping state load."
            )
