import numpy as np
from operator import sub


class levenshtein_distance:
    def __init__(
        self,
        seq1: str,
        seq2: str,
        ins_cost: int = 1,
        rep_cost: int = 1,
        del_cost: int = 1,
    ):
        self.seq1 = seq1
        self.seq2 = seq2
        self.len_seq1 = len(self.seq1)
        self.len_seq2 = len(self.seq2)

        self.ins_cost = ins_cost
        self.rep_cost = rep_cost
        self.del_cost = del_cost

        self.dist_val = self.levenshtein()

    def levenshtein(self) -> int:
        seq_dict, seq_arr = create_sequence_data(self.seq1, self.seq2)

        seq1_len = len(seq_dict["seq1"])
        seq2_len = len(seq_dict["seq2"])

        for x in range(seq1_len):
            for y in range(seq2_len):
                op_values = dynamic_operations(x, y, seq_dict, seq_arr)
                seq_arr = self.incr_seq(x, y, seq_arr, op_values)

        dist_val = int(seq_arr[self.len_seq2][self.len_seq1])
        return dist_val

    def ratio(self) -> float:
        total_lens = len(self.seq1) + len(self.seq2)
        ratio_calc = (total_lens - self.dist_val) / total_lens
        return ratio_calc

    def distance(self) -> int:
        return self.dist_val

    def ops_incr_dict(self) -> dict[str, int]:
        ops_incr = {
            "beginning": 0,
            "matching": 0,
            "insert": self.ins_cost,
            "replace": self.rep_cost,
            "delete": self.del_cost,
        }
        return ops_incr

    def incr_seq(
        self, x: int, y: int, seq_arr: np.ndarray, op_values: dict
    ) -> np.ndarray:
        op_key = op_values["key"]
        ops_incr = self.ops_incr_dict()
        ops_incr_val = ops_incr[op_key]

        op_value = op_values["val"]
        seq_arr[y][x] = op_value + ops_incr_val
        return seq_arr


def create_sequence_data(seq1: str, seq2: str) -> tuple[dict, np.ndarray]:
    sequence_dict = {}

    seq1_l, seq2_l = [*seq1], [*seq2]
    seq1_l, seq2_l = insert_null_onset(seq1_l, seq2_l)
    sequences = [seq1_l, seq2_l]

    seq1_len = len(seq1_l)
    seq2_len = len(seq2_l)

    zero_seq_arr = np.zeros((seq2_len, seq1_len))

    for s_index, seq in enumerate(sequences):
        sequence_str = f"seq{s_index+1}"
        for l_index, letter in enumerate(seq):
            if sequence_str not in sequence_dict:
                sequence_dict.update({sequence_str: {}})
            sequence_dict[sequence_str].update({l_index: letter})
    return sequence_dict, zero_seq_arr


def insert_null_onset(seq1: list, seq2: list) -> tuple[list, list]:
    seq1.insert(0, " ")
    seq2.insert(0, " ")
    return seq1, seq2


def insert_operation(x: int, y: int, seq_dict: dict) -> tuple[int, int]:
    seq1_len = len(seq_dict["seq1"])
    seq2_len = len(seq_dict["seq2"])

    insert_state = (None, sub) if seq1_len < seq2_len else (sub, None)

    x_op, y_op = insert_state

    x = x_op(x, 1) if x_op else x
    y = y_op(y, 1) if y_op else y
    return x, y


def replace_operation(x: int, y: int, seq_dict: dict) -> tuple[int, int]:
    replace_state = (sub, sub)
    x_op, y_op = replace_state

    x = x_op(x, 1) if x_op else x
    y = y_op(y, 1) if y_op else y
    return x, y


def delete_operation(x: int, y: int, seq_dict: dict) -> tuple[int, int]:
    seq1_len = len(seq_dict["seq1"])
    seq2_len = len(seq_dict["seq2"])
    delete_state = (sub, None) if seq1_len < seq2_len else (None, sub)
    x_op, y_op = delete_state

    x = x_op(x, 1) if x_op else x
    y = y_op(y, 1) if y_op else y
    return x, y


def filter_ops_dict(ops_list: list) -> list:
    filtered_ops = [op for op in ops_list if op["val"] is not None]
    return filtered_ops


def dynamic_operations(x: int, y: int, seq_dict: dict, seq_arr: np.ndarray) -> dict:
    x_dict = seq_dict["seq1"][x]
    y_dict = seq_dict["seq2"][y]

    x_ins, y_ins = insert_operation(x, y, seq_dict)
    x_rep, y_rep = replace_operation(x, y, seq_dict)
    x_del, y_del = delete_operation(x, y, seq_dict)

    ins_val = seq_arr[y_ins][x_ins] if (x_ins >= 0 and y_ins >= 0) else None
    rep_val = seq_arr[y_rep][x_rep] if (x_rep >= 0 and y_rep >= 0) else None
    del_val = seq_arr[y_del][x_del] if (x_del >= 0 and y_del >= 0) else None

    ops_list = [
        {"x": x_ins, "y": y_ins, "val": ins_val, "key": "insert"},
        {"x": x_rep, "y": y_rep, "val": rep_val, "key": "replace"},
        {"x": x_del, "y": y_del, "val": del_val, "key": "delete"},
    ]

    ops_list = filter_ops_dict(ops_list)

    if ops_list:
        value_key = "val"
        min_value = min((int(d[value_key])) for d in ops_list)
        min_ops = [k for k in ops_list if (int(k[value_key])) == min_value]

        min_op_dict = min_ops[0]

        if (x_dict is not None and y_dict is not None) and (x_dict == y_dict):
            x_m = min_op_dict["x"]
            y_m = min_op_dict["y"]
            op_dict = {"x": x_m, "y": y_m, "val": seq_arr[y_m][x_m], "key": "matching"}

        else:
            op_dict = min_op_dict

    else:
        op_dict = {"x": 0, "y": 0, "val": 0, "key": "beginning"}
    return op_dict
