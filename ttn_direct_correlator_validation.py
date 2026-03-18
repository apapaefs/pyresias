import argparse
import json
import random

import numpy as np

import pyresias_qtilde_ttn as shower


def root_node(q_tree, qbar_tree, root_label):
    return q_tree if root_label == 'q' else qbar_tree


def frontier_leaf(q_tree, qbar_tree, branch_order_map, step, root_label, path):
    node = shower.GetTreeNodeByPath(root_node(q_tree, qbar_tree, root_label), path)
    branch_key = (root_label, tuple(path))
    return node['branch'] is None or branch_order_map[branch_key] > step


def frontier_norm_matrix(q_tree, qbar_tree, branch_order_map, step, root_label, path=(), cache=None):
    if cache is None:
        cache = {}
    key = (root_label, tuple(path))
    if key in cache:
        return cache[key]
    node = shower.GetTreeNodeByPath(root_node(q_tree, qbar_tree, root_label), path)
    if frontier_leaf(q_tree, qbar_tree, branch_order_map, step, root_label, path):
        result = np.eye(shower.color_dimension(node['pid']), dtype=np.complex128)
    else:
        tensor = shower.BranchColorTensor(node['pid'], node['branch'])
        norm_left = frontier_norm_matrix(q_tree, qbar_tree, branch_order_map, step, root_label, path + (0,), cache)
        norm_right = frontier_norm_matrix(q_tree, qbar_tree, branch_order_map, step, root_label, path + (1,), cache)
        result = np.einsum('pab,ac,bd,qcd->pq', tensor, norm_left, norm_right, np.conjugate(tensor))
    cache[key] = result
    return result


def single_insertion(q_tree, qbar_tree, branch_order_map, step, root_label, target_path, same_root, current_path=(), cache=None, norm_cache=None):
    if cache is None:
        cache = {}
    if norm_cache is None:
        norm_cache = {}
    key = (root_label, tuple(current_path), tuple(target_path), bool(same_root))
    if key in cache:
        return cache[key]

    node = shower.GetTreeNodeByPath(root_node(q_tree, qbar_tree, root_label), current_path)
    if frontier_leaf(q_tree, qbar_tree, branch_order_map, step, root_label, current_path):
        if tuple(current_path) != tuple(target_path):
            raise ValueError('Reached non-target frontier leaf while building an insertion')
        generators = shower.ColorGeneratorsForPid(node['pid'])
        result = -np.transpose(generators, (0, 2, 1)) if same_root else generators
        cache[key] = result
        return result

    tensor = shower.BranchColorTensor(node['pid'], node['branch'])
    next_branch = int(target_path[len(current_path)])
    if next_branch == 0:
        inserted = single_insertion(
            q_tree, qbar_tree, branch_order_map, step, root_label, target_path, same_root, current_path + (0,), cache, norm_cache
        )
        norm_other = frontier_norm_matrix(q_tree, qbar_tree, branch_order_map, step, root_label, path=current_path + (1,), cache=norm_cache)
        result = np.einsum('pab,rac,bd,qcd->rpq', tensor, inserted, norm_other, np.conjugate(tensor))
    else:
        norm_other = frontier_norm_matrix(q_tree, qbar_tree, branch_order_map, step, root_label, path=current_path + (0,), cache=norm_cache)
        inserted = single_insertion(
            q_tree, qbar_tree, branch_order_map, step, root_label, target_path, same_root, current_path + (1,), cache, norm_cache
        )
        result = np.einsum('pab,ac,rbd,qcd->rpq', tensor, norm_other, inserted, np.conjugate(tensor))
    cache[key] = result
    return result


def double_dot(q_tree, qbar_tree, branch_order_map, step, root_label, target_path_a, target_path_b, current_path=(), cache=None, single_cache=None, norm_cache=None):
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

    if frontier_leaf(q_tree, qbar_tree, branch_order_map, step, root_label, current_path):
        raise ValueError('Reached a frontier leaf before separating the two insertion targets')

    node = shower.GetTreeNodeByPath(root_node(q_tree, qbar_tree, root_label), current_path)
    tensor = shower.BranchColorTensor(node['pid'], node['branch'])
    next_a = int(target_path_a[len(current_path)])
    next_b = int(target_path_b[len(current_path)])

    if next_a == next_b == 0:
        inserted = double_dot(
            q_tree, qbar_tree, branch_order_map, step, root_label, target_path_a, target_path_b, current_path + (0,), cache, single_cache, norm_cache
        )
        norm_other = frontier_norm_matrix(q_tree, qbar_tree, branch_order_map, step, root_label, path=current_path + (1,), cache=norm_cache)
        result = np.einsum('pab,ac,bd,qcd->pq', tensor, inserted, norm_other, np.conjugate(tensor))
    elif next_a == next_b == 1:
        norm_other = frontier_norm_matrix(q_tree, qbar_tree, branch_order_map, step, root_label, path=current_path + (0,), cache=norm_cache)
        inserted = double_dot(
            q_tree, qbar_tree, branch_order_map, step, root_label, target_path_a, target_path_b, current_path + (1,), cache, single_cache, norm_cache
        )
        result = np.einsum('pab,ac,bd,qcd->pq', tensor, norm_other, inserted, np.conjugate(tensor))
    else:
        if next_a == 0:
            inserted_left = single_insertion(
                q_tree, qbar_tree, branch_order_map, step, root_label, target_path_a, True, current_path + (0,), single_cache, norm_cache
            )
            inserted_right = single_insertion(
                q_tree, qbar_tree, branch_order_map, step, root_label, target_path_b, True, current_path + (1,), single_cache, norm_cache
            )
        else:
            inserted_left = single_insertion(
                q_tree, qbar_tree, branch_order_map, step, root_label, target_path_b, True, current_path + (0,), single_cache, norm_cache
            )
            inserted_right = single_insertion(
                q_tree, qbar_tree, branch_order_map, step, root_label, target_path_a, True, current_path + (1,), single_cache, norm_cache
            )
        result = np.einsum('pab,rac,rbd,qcd->pq', tensor, inserted_left, inserted_right, np.conjugate(tensor))
    cache[key] = result
    return result


def direct_frontier_correlators(q_tree, qbar_tree, branch_order_map, step, leaves):
    return __import__('ttn_direct_frontier').frontier_pair_correlators(shower, q_tree, qbar_tree, branch_order_map, step, leaves)


def summarize_delta(reference, trial):
    ref_map = {tuple(entry['pair']): float(entry['value']) for entry in reference}
    trial_map = {tuple(entry['pair']): float(entry['value']) for entry in trial}
    deltas = [abs(ref_map[pair] - trial_map[pair]) for pair in sorted(ref_map.keys())]
    if not deltas:
        return {'pair_count': 0, 'mean_abs_delta': 0.0, 'max_abs_delta': 0.0}
    return {
        'pair_count': int(len(deltas)),
        'mean_abs_delta': float(sum(deltas) / len(deltas)),
        'max_abs_delta': float(max(deltas)),
    }


def main():
    parser = argparse.ArgumentParser(description='Validate direct TTN color correlators against dense correlators on frontier slices.')
    parser.add_argument('inputfile')
    parser.add_argument('-n', '--n-events', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--max-gluons', type=int, default=6)
    parser.add_argument('--json-out', default='')
    args = parser.parse_args()

    random.seed(args.seed)
    shower.seed(args.seed)
    shower.configure_from_args(['-n', str(args.n_events), '--skip-output', '--ttn-max-gluons', str(args.max_gluons), args.inputfile])
    a_s_over = shower.get_alphaS_over(shower.Qc)
    events, _, _ = shower.readlhefile(shower.inputfile)

    checked_slices = 0
    mean_deltas = []
    max_delta = 0.0
    records = []

    for event_index, particles in enumerate(events[: args.n_events]):
        _, _, histories = shower.Shower(particles, shower.pTmin, a_s_over)
        q_hist = [history for history in histories if history['branch_type'] == 'q'][0]
        qbar_hist = [history for history in histories if history['branch_type'] == 'qbar'][0]
        order_map, branch_records = shower.BuildBranchOrderMap(q_hist['tree'], qbar_hist['tree'])
        for step in range(len(branch_records) + 1):
            leaves = shower.CollectFrontierLeaves(q_hist['tree'], 'q', order_map, step) + shower.CollectFrontierLeaves(qbar_hist['tree'], 'qbar', order_map, step)
            for axis, leaf in enumerate(leaves):
                leaf['axis'] = axis
            frontier_dimension = shower.DenseStateDimensionFromLeaves(leaves)
            if frontier_dimension > shower.DenseDimensionCap(shower.ttn_max_gluons):
                continue
            state, norm = shower.BuildFrontierColorState(q_hist['tree'], qbar_hist['tree'], order_map, step)
            if norm <= shower.TTN_EPS:
                continue
            dense = shower.AllPairColorCorrelators(state, leaves)
            direct = direct_frontier_correlators(q_hist['tree'], qbar_hist['tree'], order_map, step, leaves)
            summary = summarize_delta(dense, direct)
            checked_slices += 1
            mean_deltas.append(summary['mean_abs_delta'])
            max_delta = max(max_delta, summary['max_abs_delta'])
            records.append({
                'event_index': int(event_index),
                'step': int(step),
                'active_line_count': int(len(leaves)),
                'frontier_dimension': int(frontier_dimension),
                'mean_abs_delta': float(summary['mean_abs_delta']),
                'max_abs_delta': float(summary['max_abs_delta']),
            })

    payload = {
        'checked_slices': int(checked_slices),
        'mean_of_mean_abs_delta': float(sum(mean_deltas) / len(mean_deltas)) if mean_deltas else 0.0,
        'max_abs_delta': float(max_delta),
        'records': records,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out:
        with open(args.json_out, 'w') as fout:
            json.dump(payload, fout, indent=2, sort_keys=True)
            fout.write('\n')


if __name__ == '__main__':
    main()
