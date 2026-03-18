import numpy as np


def _root_node(q_tree, qbar_tree, root_label):
    return q_tree if root_label == "q" else qbar_tree


def frontier_norm_matrix(shower, q_tree, qbar_tree, branch_order_map, step, root_label, path=(), cache=None):
    if cache is None:
        cache = {}
    key = (root_label, tuple(path))
    if key in cache:
        return cache[key]

    node = shower.GetTreeNodeByPath(_root_node(q_tree, qbar_tree, root_label), path)
    branch_key = (root_label, tuple(path))
    if node["branch"] is None or branch_order_map[branch_key] > step:
        result = np.eye(shower.color_dimension(node["pid"]), dtype=np.complex128)
    else:
        tensor = shower.BranchColorTensor(node["pid"], node["branch"])
        norm_left = frontier_norm_matrix(shower, q_tree, qbar_tree, branch_order_map, step, root_label, path + (0,), cache)
        norm_right = frontier_norm_matrix(shower, q_tree, qbar_tree, branch_order_map, step, root_label, path + (1,), cache)
        result = np.einsum("pab,ac,bd,qcd->pq", tensor, norm_left, norm_right, np.conjugate(tensor))
    cache[key] = result
    return result


def frontier_single_insertion(
    shower, q_tree, qbar_tree, branch_order_map, step, root_label, target_path, path=(), dual=False, cache=None, norm_cache=None
):
    if cache is None:
        cache = {}
    if norm_cache is None:
        norm_cache = {}
    key = (root_label, tuple(path), tuple(target_path), bool(dual))
    if key in cache:
        return cache[key]

    node = shower.GetTreeNodeByPath(_root_node(q_tree, qbar_tree, root_label), path)
    branch_key = (root_label, tuple(path))
    if node["branch"] is None or branch_order_map[branch_key] > step:
        if tuple(path) != tuple(target_path):
            raise ValueError("Reached non-target frontier leaf while building a color insertion")
        generators = shower.ColorGeneratorsForPid(node["pid"])
        result = -np.transpose(generators, (0, 2, 1)) if dual else generators
        cache[key] = result
        return result

    tensor = shower.BranchColorTensor(node["pid"], node["branch"])
    next_branch = int(target_path[len(path)])
    if next_branch == 0:
        inserted = frontier_single_insertion(
            shower, q_tree, qbar_tree, branch_order_map, step, root_label, target_path, path + (0,), dual, cache, norm_cache
        )
        norm_other = frontier_norm_matrix(shower, q_tree, qbar_tree, branch_order_map, step, root_label, path + (1,), norm_cache)
        result = np.einsum("pab,rac,bd,qcd->rpq", tensor, inserted, norm_other, np.conjugate(tensor))
    else:
        norm_other = frontier_norm_matrix(shower, q_tree, qbar_tree, branch_order_map, step, root_label, path + (0,), norm_cache)
        inserted = frontier_single_insertion(
            shower, q_tree, qbar_tree, branch_order_map, step, root_label, target_path, path + (1,), dual, cache, norm_cache
        )
        result = np.einsum("pab,ac,rbd,qcd->rpq", tensor, norm_other, inserted, np.conjugate(tensor))
    cache[key] = result
    return result


def frontier_double_dot(
    shower,
    q_tree,
    qbar_tree,
    branch_order_map,
    step,
    root_label,
    target_path_a,
    target_path_b,
    path=(),
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
    key = (root_label, tuple(path), ordered_targets)
    if key in cache:
        return cache[key]

    node = shower.GetTreeNodeByPath(_root_node(q_tree, qbar_tree, root_label), path)
    branch_key = (root_label, tuple(path))
    if node["branch"] is None or branch_order_map[branch_key] > step:
        raise ValueError("Reached a frontier leaf before separating the two insertion targets")

    tensor = shower.BranchColorTensor(node["pid"], node["branch"])
    next_a = int(target_path_a[len(path)])
    next_b = int(target_path_b[len(path)])
    if next_a == next_b == 0:
        inserted = frontier_double_dot(
            shower,
            q_tree,
            qbar_tree,
            branch_order_map,
            step,
            root_label,
            target_path_a,
            target_path_b,
            path + (0,),
            cache,
            single_cache,
            norm_cache,
        )
        norm_other = frontier_norm_matrix(shower, q_tree, qbar_tree, branch_order_map, step, root_label, path + (1,), norm_cache)
        result = np.einsum("pab,ac,bd,qcd->pq", tensor, inserted, norm_other, np.conjugate(tensor))
    elif next_a == next_b == 1:
        norm_other = frontier_norm_matrix(shower, q_tree, qbar_tree, branch_order_map, step, root_label, path + (0,), norm_cache)
        inserted = frontier_double_dot(
            shower,
            q_tree,
            qbar_tree,
            branch_order_map,
            step,
            root_label,
            target_path_a,
            target_path_b,
            path + (1,),
            cache,
            single_cache,
            norm_cache,
        )
        result = np.einsum("pab,ac,bd,qcd->pq", tensor, norm_other, inserted, np.conjugate(tensor))
    else:
        if next_a == 0:
            inserted_left = frontier_single_insertion(
                shower, q_tree, qbar_tree, branch_order_map, step, root_label, target_path_a, path + (0,), True, single_cache, norm_cache
            )
            inserted_right = frontier_single_insertion(
                shower, q_tree, qbar_tree, branch_order_map, step, root_label, target_path_b, path + (1,), True, single_cache, norm_cache
            )
        else:
            inserted_left = frontier_single_insertion(
                shower, q_tree, qbar_tree, branch_order_map, step, root_label, target_path_b, path + (0,), True, single_cache, norm_cache
            )
            inserted_right = frontier_single_insertion(
                shower, q_tree, qbar_tree, branch_order_map, step, root_label, target_path_a, path + (1,), True, single_cache, norm_cache
            )
        result = np.einsum("pab,rac,rbd,qcd->pq", tensor, inserted_left, inserted_right, np.conjugate(tensor))
    cache[key] = result
    return result


def _orientation_flip(shower, root_node, path):
    node = root_node
    flip = False
    for child_idx in path:
        branch = node["branch"]
        branch_type = branch["type"]
        child_idx = int(child_idx)
        if branch_type == "q->qg" and child_idx == 1:
            flip = not flip
        elif branch_type == "g->qqbar":
            flip = not flip
        node = node["children"][child_idx]
    return flip


def _cross_insert(shower, q_tree, qbar_tree, branch_order_map, step, root_label, path, single_cache, norm_cache):
    base = frontier_single_insertion(
        shower, q_tree, qbar_tree, branch_order_map, step, root_label, tuple(path), (), False, single_cache, norm_cache
    )
    root_node = _root_node(q_tree, qbar_tree, root_label)
    if not _orientation_flip(shower, root_node, tuple(path)):
        return base
    return -np.transpose(base, (0, 2, 1))


def frontier_pair_correlators(shower, q_tree, qbar_tree, branch_order_map, step, leaves):
    norm_cache = {}
    single_cache = {}
    double_cache = {}
    norm_q = frontier_norm_matrix(shower, q_tree, qbar_tree, branch_order_map, step, "q", (), norm_cache)
    norm_qbar = frontier_norm_matrix(shower, q_tree, qbar_tree, branch_order_map, step, "qbar", (), norm_cache)
    norm_sq = float(np.real(np.sum(norm_q * norm_qbar) / 3.0))
    if norm_sq <= shower.TTN_EPS:
        return tuple()

    correlators = []
    for axis_i in range(len(leaves)):
        for axis_j in range(axis_i + 1, len(leaves)):
            leaf_i = leaves[axis_i]
            leaf_j = leaves[axis_j]
            if leaf_i["root"] == leaf_j["root"]:
                pair_matrix = frontier_double_dot(
                    shower,
                    q_tree,
                    qbar_tree,
                    branch_order_map,
                    step,
                    leaf_i["root"],
                    tuple(leaf_i["path"]),
                    tuple(leaf_j["path"]),
                    (),
                    double_cache,
                    single_cache,
                    norm_cache,
                )
                env = norm_qbar if leaf_i["root"] == "q" else norm_q
                value = np.sum(pair_matrix * env) / (3.0 * norm_sq)
            else:
                q_leaf = leaf_i if leaf_i["root"] == "q" else leaf_j
                qbar_leaf = leaf_j if leaf_i["root"] == "q" else leaf_i
                inserted_q = _cross_insert(shower, q_tree, qbar_tree, branch_order_map, step, "q", q_leaf["path"], single_cache, norm_cache)
                inserted_qbar = _cross_insert(
                    shower, q_tree, qbar_tree, branch_order_map, step, "qbar", qbar_leaf["path"], single_cache, norm_cache
                )
                value = np.sum(inserted_q * inserted_qbar) / (3.0 * norm_sq)
            correlators.append({"pair": (int(axis_i), int(axis_j)), "value": float(np.real(value))})
    return tuple(correlators)


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
