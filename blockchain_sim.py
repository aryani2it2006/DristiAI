"""
engine/blockchain_sim.py

*** DEMO STAND-IN ONLY -- NOT THE SPEC-MANDATED BLOCKCHAIN ***

Sections 54-61 call for a real private EVM-compatible blockchain, a
Vyper smart contract, and web3.py as the client library. Building that
needs an actual chain node (e.g. a local Ganache/Anvil instance) running
alongside this code, plus the web3.py package -- neither of which is
available in the environment I tested this in (no network to install
web3.py, no chain node running).

To let you see the CONCEPT end-to-end right now (Section 59's workflow:
event -> evidence hash -> block -> chained to previous block -> stored),
this file implements a minimal in-process SHA-256 hash-chain: each
"block" commits to the previous block's hash, exactly like a real chain
does, but with no smart contract, no accounts, and no real EVM.

When we reach Phase 11 (Section 81) for real, this file gets replaced by
an actual Vyper contract + web3.py client talking to a local chain --
we'll build that as its own module, not patch this one.
"""

import hashlib
import json
from datetime import datetime, timezone


class HashChainLedger:
    def __init__(self):
        self.chain = []
        self._add_genesis_block()

    def _add_genesis_block(self):
        genesis = {
            "index": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": None,
            "evidence_hash": None,
            "previous_hash": "0" * 64,
        }
        genesis["block_hash"] = self._compute_hash(genesis)
        self.chain.append(genesis)

    @staticmethod
    def _compute_hash(block_without_hash):
        payload = json.dumps(block_without_hash, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()

    def add_record(self, event_id, evidence_hash):
        previous_hash = self.chain[-1]["block_hash"]
        block = {
            "index": len(self.chain),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": event_id,
            "evidence_hash": evidence_hash,
            "previous_hash": previous_hash,
        }
        block["block_hash"] = self._compute_hash(block)
        self.chain.append(block)
        return block

    def find_by_event(self, event_id):
        for block in self.chain:
            if block["event_id"] == event_id:
                return block
        return None

    def verify_chain(self):
        """Section 60 -- Blockchain Verification Workflow (simplified)."""
        for i in range(1, len(self.chain)):
            block = dict(self.chain[i])
            stored_hash = block.pop("block_hash")
            recomputed = self._compute_hash(block)
            if recomputed != stored_hash:
                return False, f"Block {i} hash mismatch (tampered)"
            if block["previous_hash"] != self.chain[i - 1]["block_hash"]:
                return False, f"Block {i} broken link to previous block"
        return True, "Chain intact"

    def verify_event(self, event_id, evidence_file_hash):
        """
        Section 60: does the evidence file's current hash still match what
        was committed to the ledger for this event?
        """
        block = self.find_by_event(event_id)
        if block is None:
            return {"found": False, "match": False, "reason": "event not on ledger"}
        match = block["evidence_hash"] == evidence_file_hash
        return {"found": True, "match": match, "block": block}
