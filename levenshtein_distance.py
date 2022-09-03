import numpy as np
from operator import sub


def create_sequence_data(seq1: str, seq2: str) -> tuple[dict, np.ndarray]:
    sequence_dict = {}

    seq1_l, seq2_l = [*seq1], [*seq2]
    seq1_l, seq2_l = insert_null_onset(seq1_l, seq2_l)
    sequences = [seq1_l, seq2_l]

    max_seq = max([seq1_l, seq2_l], key=len)
    max_seq_len = len(max_seq)

    zero_seq_arr = np.zeros((max_seq_len, max_seq_len))

    for s_index, seq in enumerate(sequences):
        sequence_str = f"seq{s_index}"
        final_index = 0
        for l_index, letter in enumerate(seq):
            final_index = l_index
            if sequence_str not in sequence_dict:
                sequence_dict.update({sequence_str: {}})
            sequence_dict[sequence_str].update({l_index: letter})

        if len(seq) < max_seq_len:
            diff = max_seq_len - len(seq)
            for _ in range(diff):
                final_index += 1
                sequence_dict[sequence_str].update({final_index: ""})

    return sequence_dict, zero_seq_arr


def insert_null_onset(seq1, seq2):
    seq1.insert(0, ' ')
    seq2.insert(0, ' ')
    return seq1, seq2

def insert_operation(x: int, y: int) -> tuple[int, int]:
    insert_state = (None, sub)

    x_op, y_op = insert_state

    x = x_op(x, 1) if x_op else x
    y = y_op(y, 1) if y_op else y

    return x, y


def replace_operation(x: int, y: int) -> tuple[int, int]:
    replace_state = (sub, sub)
    x_op, y_op = replace_state

    x = x_op(x, 1) if x_op else x
    y = y_op(y, 1) if y_op else y

    return x, y


def delete_operation(x: int, y: int) -> tuple[int, int]:
    delete_state = (sub, None)
    x_op, y_op = delete_state

    x = x_op(x, 1) if x_op else x
    y = y_op(y, 1) if y_op else y

    return x, y


def filter_ops_dict(ops_dict: dict) -> dict:
    keys_pop = []
    for key, values in ops_dict.items():
        if values["val"] is None:
            keys_pop.append(key)
    for key in keys_pop:
        ops_dict.pop(key)
    return ops_dict


def dynamic_operations(x: int, y: int, seq_dict: dict, seq_arr: np.ndarray) -> dict:
    x_dict = seq_dict["seq0"][x]
    y_dict = seq_dict["seq1"][y]

    x_ins, y_ins = insert_operation(x, y)
    x_rep, y_rep = replace_operation(x, y)
    x_del, y_del = delete_operation(x, y)

    ins_val = seq_arr[y_ins][x_ins] if (x_ins >= 0 and y_ins >= 0) else None
    rep_val = seq_arr[y_rep][x_rep] if (x_rep >= 0 and y_rep >= 0) else None
    del_val = seq_arr[y_del][x_del] if (x_del >= 0 and y_del >= 0) else None

    ops_dict = {
        "insert": {"x": x_ins, "y": y_ins, "val": ins_val, "key": "insert"},
        "replace": {"x": x_rep, "y": y_rep, "val": rep_val, "key": "replace"},
        "delete": {"x": x_del, "y": y_del, "val": del_val, "key": "delete"},
    }

    ops_dict = filter_ops_dict(ops_dict)

    if ops_dict:
        value_key = "val"
        lowest_value = min((int(d[value_key])) for d in ops_dict.values())
        lowest_key = [
            k for k in ops_dict if (int(ops_dict[k][value_key])) == lowest_value
        ]

        if x_dict == y_dict:
            x_m = ops_dict[lowest_key[0]]["x"]
            y_m = ops_dict[lowest_key[0]]["y"]
            op_values = {
                "x": x_m,
                "y": y_m,
                "val": seq_arr[y_m][x_m],
                "key": "matching",
            }

        else:
            op_values = ops_dict[lowest_key[0]]

    else:
        op_values = {"x": 0, "y": 0, "val": 0, "key": "beginning"}

    return op_values


def ops_incr_dict():
    ops_incr = {
        "beginning" : 0,
        "matching" : 0,
        "insert" : 1,
        "replace" : 1,
        "delete" : 1,
    }
    return ops_incr


def incr_seq(x: int, y: int, seq_arr: np.ndarray, op_values: dict) -> np.ndarray:
    op_key = op_values["key"]
    ops_incr = ops_incr_dict()
    ops_incr_val = ops_incr[op_key]

    op_value = op_values["val"]
    seq_arr[y][x] = op_value + ops_incr_val

    return seq_arr


def levenshtein(seq1: str, seq2: str) -> int:
    seq_dict, seq_arr = create_sequence_data(seq1, seq2)

    len_arr = len(seq_arr[0])

    for x in range(len_arr):
        for y in range(len_arr):
            op_values = dynamic_operations(x, y, seq_dict, seq_arr)
            seq_arr = incr_seq(x, y, seq_arr, op_values)

    max_len_1 = len(seq_dict["seq0"]) - 1
    max_len_2 = len(seq_dict["seq1"]) - 1
    dist_val = int(seq_arr[max_len_1][max_len_2])

    return dist_val
