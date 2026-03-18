import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import pyresias_qtilde_ttn as shower


def _root_node(q_tree, qbar_tree, root_label):
    return q_tree if root_label == "q" else qbar_tree


def _frontier_leaf(q_tree, qbar_tree, branch_order_map, step, root_label, path):
    node = shower.GetTreeNodeByPath(_root_node(q_tree, qbar_tree, root_label), path)
    branch_key = (root_label, tuple(path))
    return node["branch"] is None or branch_order_map[branch_key] > step


def frontier_norm_matrix(q_tree, qbar_tree, branch_order_map, step, root_label, path=(), cache=None):
    if cache is None:
        cache = {}
    key = (root_label, tuple(path))
    if key in cache:
        return cache[key]

    node = shower.GetTreeNodeByPath(_root_node(q_tree, qbar_tree, root_label), path)
    if _frontier_leaf(q_tree, qbar_tree, branch_order_map, step, root_label, path):
        result = np.eye(shower.color_dimension(node["pid"]), dtype=np.complex128)
    else:
        tensor = shower.BranchColorTensor(node["pid"], node["branch"])
        norm_left = frontier_norm_matrix(q_tree, qbar_tree, branch_order_map, step, root_label, path + (0,), cache)
        norm_right = frontier_norm_matrix(q_tree, qbar_tree, branch_order_map, step, root_label, path + (1,), cache)
        result = np.einsum("pab,ac,bd,qcd->pq", tensor, norm_left, norm_right, np.conjugate(tensor))
    cache[key] = result
    return result


def frontier_single_insertion(q_tree, qbar_tree, branch_order_map, step, root_label, target_path, current_path=(), cache=None, norm_cache=None):
    if cache is None:
        cache = {}
    if norm_cache is None:
        norm_cache = {}
    key = (root_label, tuple(current_path), tuple(target_path))
    if key in cache:
        return cache[key]

    node = shower.GetTreeNodeByPath(_root_node(q_tree, qbar_tree, root_label), current_path)
    if _frontier_leaf(q_tree, qbar_tree, branch_order_map, step, root_label, current_path):
        if tuple(current_path) != tuple(target_path):
            raise ValueError("Reached non-target frontier leaf while building a single insertion")
        result = shower.ColorGeneratorsForPid(node["pid"])
        cache[key] = result
        return result

    tensor = shower.BranchColorTensor(node["pid"], node["branch"])
    next_branch = int(target_path[len(current_path)])
    if next_branch == 0:
        inserted = frontier_single_insertion(
            q_tree, qbar_tree, branch_order_map, step, root_label, target_path, current_path + (0,), cache, norm_cache
        )
        norm_other = frontier_norm_matrix(q_tree, qbar_tree, branch_order_map, step, root_label, current_path + (1,), norm_cache)
        result = np.einsum("pab,rac,bd,qcd->rpq", tensor, inserted, norm_other, np.conjugate(tensor))
    else:
        norm_other = frontier_norm_matrix(q_tree, qbar_tree, branch_order_map, step, root_label, current_path + (0,), norm_cache)
        inserted = frontier_single_insertion(
            q_tree, qbar_tree, branch_order_map, step, root_label, target_path, current_path + (1,), cache, norm_cache
        )
        result = np.einsum("pab,ac,rbd,qcd->rpq", tensor, norm_other, inserted, np.conjugate(tensor))
    cache[key] = result
    return result


def frontier_double_dot(
    q_tree,
    qbar_tree,
    branch_order_map,
    step,
    root_label,
    target_path_a,
    target_path_b,
    current_path=(),
    cache=None,
    single_cache=None,
    norm_cache=None,
):
    if cache is None:
        cache = {}
    if single_cache is None:
        single_cache = {}
    if norm_cache is None:
        norm_cache = {}
    ordered_targets = tuple(sorted((tuple(target_path_a), tuple(target_path_b))))
    key = (root_label, tuple(current_path), ordered_targets)
    if key in cache:
        return cache[key]

    if _frontier_leaf(q_tree, qbar_tree, branch_order_map, step, root_label, current_path):
        raise ValueError("Reached a frontier leaf before separating the two insertion targets")

    node = shower.GetTreeNodeByPath(_root_node(q_tree, qbar_tree, root_label), current_path)
    tensor = shower.BranchColorTensor(node["pid"], node["branch"])
    next_a = int(target_path_a[len(current_path)])
    next_b = int(target_path_b[len(current_path)])

    if next_a == next_b == 0:
        inserted = frontier_double_dot(
            q_tree,
            qbar_tree,
            branch_order_map,
            step,
            root_label,
            target_path_a,
            target_path_b,
            current_path + (0,),
            cache,
            single_cache,
            norm_cache,
        )
        norm_other = frontier_norm_matrix(q_tree, qbar_tree, branch_order_map, step, root_label, current_path + (1,), norm_cache)
        result = np.einsum("pab,ac,bd,qcd->pq", tensor, inserted, norm_other, np.conjugate(tensor))
    elif next_a == next_b == 1:
        norm_other = frontier_norm_matrix(q_tree, qbar_tree, branch_order_map, step, root_label, current_path + (0,), norm_cache)
        inserted = frontier_double_dot(
            q_tree,
            qbar_tree,
            branch_order_map,
            step,
            root_label,
            target_path_a,
            target_path_b,
            current_path + (1,),
            cache,
            single_cache,
            norm_cache,
        )
        result = np.einsum("pab,ac,bd,qcd->pq", tensor, norm_other, inserted, np.conjugate(tensor))
    else:
        if next_a == 0:
            inserted_left = frontier_single_insertion(
                q_tree, qbar_tree, branch_order_map, step, root_label, target_path_a, current_path + (0,), single_cache, norm_cache
            )
            inserted_right = frontier_single_insertion(
                q_tree, qbar_tree, branch_order_map, step, root_label, target_path_b, current_path + (1,), single_cache, norm_cache
            )
        else:
            inserted_left = frontier_single_insertion(
                q_tree, qbar_tree, branch_order_map, step, root_label, target_path_b, current_path + (0,), single_cache, norm_cache
            )
            inserted_right = frontier_single_insertion(
                q_tree, qbar_tree, branch_order_map, step, root_label, target_path_a, current_path + (1,), single_cache, norm_cache
            )
        result = np.einsum("pab,rac,rbd,qcd->pq", tensor, inserted_left, inserted_right, np.conjugate(tensor))
    cache[key] = result
    return result


def frontier_pair_correlators_direct(q_tree, qbar_tree, branch_order_map, step, leaves):
    return __import__("ttn_direct_frontier").frontier_pair_correlators(shower, q_tree, qbar_tree, branch_order_map, step, leaves)


def summarize_correlator_delta(reference, trial):
    ref_map = {tuple(entry["pair"]): float(entry["value"]) for entry in reference}
    trial_map = {tuple(entry["pair"]): float(entry["value"]) for entry in trial}
    deltas = [abs(ref_map[pair] - trial_map[pair]) for pair in sorted(ref_map.keys())]
    if not deltas:
        return {"pair_count": 0, "mean_abs_delta": 0.0, "max_abs_delta": 0.0}
    return {
        "pair_count": int(len(deltas)),
        "mean_abs_delta": float(sum(deltas) / len(deltas)),
        "max_abs_delta": float(max(deltas)),
    }
