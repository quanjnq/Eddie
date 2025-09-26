import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import random
import itertools
from util.const_util  import data_gen_work_dir, RANDOMWALK_INDCFG

def random_walk(candidates, visited, max_steps=5):
    # Initialize the random-walk record
    walk_steps = []  # Used to record the path of each step

    # The first step is to randomly select an element from the candidate set
    current = random.choice(candidates)
    visited.add(current)
    walk = [current]

    walk_steps.append(walk.copy())  # Save the current path

    for step in range(1, max_steps):
        # Each step has a 50% probability of termination
        if random.random() > 0.5:
            break

        # Select Next Element (Non-Repeat)
        next_candidates = [x for x in candidates if x not in visited]
        if not next_candidates:
            break  # If there are no unvisited elements, they are terminated

        next_step = random.choice(next_candidates)
        visited.add(next_step)
        walk.append(next_step)

        walk_steps.append(walk.copy())  # Save the current path

    return walk_steps



def get_indcfg_list(query,  visted, sample_multiple=2):
    indexable_columns = set(query.columns)
    cand_inds = set()
    cols = set()
    tbl2cols = {}
    imdb_black_index_cols = ["movie_info.info", "movie_info_idx.note", "person_info.info"] # The field is too large for PG to index
    for col in indexable_columns:
        if f"{col.table.name}.{col.name}" in imdb_black_index_cols:
            continue
        if col.table.name not in tbl2cols:
            tbl2cols[col.table.name] = []
        tbl2cols[col.table.name].append(col.name)
        cand_inds.add(f"{col.table.name}#{col.name}")
    
    for tbl in tbl2cols:
        if len(tbl2cols[tbl]) < 2:
            continue
        ind2 = list(itertools.permutations(tbl2cols[tbl], 2))
        for cols in ind2:
            ind = f"{tbl}#{cols[0]},{cols[1]}"
            cand_inds.add(ind)
    tgt_num = len(visted) * sample_multiple # Target the number of samples to be expanded
    indcfg_list = []
    if len(cand_inds) == 0:
        return []
    max_loop = 100
    cnt_loop = 0
    while len(indcfg_list) < tgt_num:
        walk_steps = random_walk(list(cand_inds), visted)
        for ws in walk_steps:
            indcfg = tuple(sorted(ws))
            if indcfg not in visted:
                indcfg_list.append(indcfg)
        cnt_loop += 1
        if cnt_loop > max_loop:
            break
    indcfg_list = indcfg_list[:tgt_num]
    return indcfg_list


def get_q2inds(fp):
    with open(fp, "rb") as f:
        src_data = pickle.load(f)
    q2inds = {}
    for sample in src_data:
        query = sample[0]
        init_index_comb, index_comb = sample[1]
        if query not in q2inds:
            q2inds[query] = set()
        q2inds[query].add(tuple(sorted(list(index_comb))))
    return q2inds


def enhance_index_for_dataset(dataset_path):
    print(f"Strat enhance index for dataset: {dataset_path}")
    random.seed(0)
    q_ics_pairs = []
    q2inds = get_q2inds(fp=dataset_path)

    for query in q2inds:
        ics = get_indcfg_list(query, q2inds[query])
        q_ics_pairs.append({"query": query, RANDOMWALK_INDCFG: ics})

    fn = dataset_path.split("/")[-1].split(".")[0]
    save_path = data_gen_work_dir + f"candidate_indexes/{fn}_randomwalk_enhanced.pickle"
    dst_dir = os.path.dirname(save_path)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    with open(save_path, "wb") as f:
        pickle.dump(q_ics_pairs, f)
        print(f"Saved randomwalk enhanced candidate indexes to: {save_path}")
    return save_path


if __name__ == "__main__":
    fp = ""
    enhance_index_for_dataset(fp)