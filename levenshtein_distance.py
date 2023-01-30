from dataclasses import dataclass
from typing import Optional


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


class LevenshteinBase:
    def __init__(self, seq1: str, seq2: str):
        self._seq1 = seq1
        self._seq2 = seq2
        self._ins_cost = 1
        self._rep_cost = 1
        self._del_cost = 1
        self._levenshtein_data: Optional[LevenshteinData] = None

    def _get_levenshtein_data(self) -> LevenshteinData:
        if not self._levenshtein_data:
            ops_costs = OpsCosts(
                ins_cost=self._ins_cost,
                rep_cost=self._rep_cost,
                del_cost=self._del_cost,
            )
            self._levenshtein_data = self._calculate_distance(
                self._seq1, self._seq2, ops_costs
            )
        return self._levenshtein_data

    def _calculate_distance(
        self, seq1: str, seq2: str, ops_costs: OpsCosts
    ) -> LevenshteinData:
        seq_op = self._create_sequence_data(seq1, seq2)
        seq1_len = seq_op.seq1_len
        seq2_len = seq_op.seq2_len

        for x in range(seq1_len):
            for y in range(seq2_len):
                op_values = self._dynamic_operations(x, y, seq_op)
                op_value = op_values["val"]
                op_key = op_values["key"]
                op_cost = ops_costs.as_dict()[op_key]
                seq_op.seq_arr[y][x] = op_value + op_cost

        seq_arr = seq_op.seq_arr
        dist_val = int(seq_arr[seq2_len - 1][seq1_len - 1])
        lev_data = LevenshteinData(distance=dist_val, seq_arr=seq_arr)
        return lev_data

    def _create_sequence_data(self, seq1: str, seq2: str) -> SeqOp:
        seq_dict: dict[str, dict[int, str]] = {}

        seq1_l, seq2_l = self._insert_null_onset([*seq1], [*seq2])
        seq1_len, seq2_len = len(seq1_l), len(seq2_l)

        seq_arr = [[0 for _ in range(seq1_len)] for _ in range(seq2_len)]

        for s_index, seq in enumerate([seq1_l, seq2_l]):
            sequence_str = f"seq{s_index+1}"
            for l_index, letter in enumerate(seq):
                if sequence_str not in seq_dict:
                    seq_dict.update({sequence_str: {}})
                seq_dict[sequence_str].update({l_index: letter})

        seq_op = SeqOp(
            seq_arr=seq_arr, seq_dict=seq_dict, seq1_len=seq1_len, seq2_len=seq2_len
        )
        return seq_op

    @staticmethod
    def _insert_null_onset(seq1: list, seq2: list) -> tuple[list, list]:
        seq1.insert(0, None)
        seq2.insert(0, None)
        return seq1, seq2

    @staticmethod
    def _get_min_ops(ops_list) -> dict:
        min_ops = None
        for op in ops_list:
            if op["x"] >= 0 and op["y"] >= 0:
                if min_ops is None:
                    min_ops = op
                    continue
                if op["val"] < min_ops["val"]:
                    min_ops = op

        if not min_ops:
            raise Exception("Failed to retrieve Minimum Operation.")
        return min_ops

    def _dynamic_operations(self, x: int, y: int, seq_op: SeqOp) -> dict:
        x_dict = seq_op.seq_dict["seq1"][x]
        y_dict = seq_op.seq_dict["seq2"][y]

        x_ins, y_ins = (x, y - 1) if seq_op.seq1_len < seq_op.seq2_len else (x - 1, y)
        x_rep, y_rep = (x - 1, y - 1)
        x_del, y_del = (x - 1, y) if seq_op.seq1_len < seq_op.seq2_len else (x, y - 1)

        ins_val = seq_op.seq_arr[y_ins][x_ins]
        rep_val = seq_op.seq_arr[y_rep][x_rep]
        del_val = seq_op.seq_arr[y_del][x_del]

        onset_state = x_rep + y_rep == -2
        match_state = x_dict == y_dict

        if onset_state:
            op_dict = {"x": 0, "y": 0, "val": 0, "key": "onset"}

        elif match_state:
            op_dict = {"x": x_rep, "y": y_rep, "val": rep_val, "key": "match"}

        else:
            ops_list = [
                {"x": x_ins, "y": y_ins, "val": ins_val, "key": "insert"},
                {"x": x_rep, "y": y_rep, "val": rep_val, "key": "replace"},
                {"x": x_del, "y": y_del, "val": del_val, "key": "delete"},
            ]

            op_dict = self._get_min_ops(ops_list)
        return op_dict


class Levenshtein(LevenshteinBase):
    def __init__(self, seq1: str, seq2: str):
        super().__init__(seq1, seq2)

    def set_insert_cost(self, cost: int):
        self._ins_cost = cost
        return self

    def set_replace_cost(self, cost: int):
        self._rep_cost = cost
        return self

    def set_delete_cost(self, cost: int):
        self._del_cost = cost
        return self

    def distance(self) -> int:
        levenshtein_data = self._get_levenshtein_data()
        return levenshtein_data.distance

    def ratio(self) -> float:
        total_lens = len(self._seq1) + len(self._seq2)
        dist_val = self.distance()
        ratio_calc = (total_lens - dist_val) / total_lens
        return ratio_calc

    def sequence_array(self) -> list:
        levenshtein_data = self._get_levenshtein_data()
        return levenshtein_data.seq_arr
