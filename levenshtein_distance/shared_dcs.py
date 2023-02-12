from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LevenshteinData:
    distance: int
    seq_arr: list


@dataclass
class SeqOp:
    seq_arr: list
    seq_dict: dict
    seq1_len: int
    seq2_len: int


@dataclass
class OpsCosts:
    onset_cost: int = 0
    match_cost: int = 0
    ins_cost: int = 1
    rep_cost: int = 1
    del_cost: int = 1

    def as_dict(self) -> dict[str, int]:
        if not hasattr(self, "ops_costs_dict"):
            self.ops_costs_dict = {
                "onset": self.onset_cost,
                "match": self.match_cost,
                "insert": self.ins_cost,
                "replace": self.rep_cost,
                "delete": self.del_cost,
            }
        return self.ops_costs_dict
