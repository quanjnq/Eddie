def eq_selectivity(value, col_stats):
    if value in col_stats["most_common_vals"]:
        return col_stats["most_common_freqs"][col_stats["most_common_vals"].index(value)]
    else:
        if (col_stats["n_distinct"] - len(col_stats["most_common_freqs"])) == 0:
            return 1e-7
        return max((1 - sum(col_stats["most_common_freqs"]) - float(col_stats["null_frac"])) / (col_stats["n_distinct"] - len(col_stats["most_common_freqs"])), 1e-7)


def lt_selectivity(value, col_stats):
    freqs = col_stats["most_common_freqs"]
    vals = col_stats["most_common_vals"]
    histogram_bounds = col_stats["histogram_bounds"]
    relevantmvfsum = 0
    for i, v in enumerate(vals):
        if v < value:
            relevantmvfsum += freqs[i]
    if histogram_bounds is None:
        return relevantmvfsum
    hist = None
    for i, v in enumerate(histogram_bounds):
        if v > value:
            if i < 1:
                # If the bucket is not hit, record a small value as the selectivity
                hist = 1e-7
            else:
                if str == type(value):
                    hist = (i-1 + 0.5) / (len(histogram_bounds) - 1)
                else:
                    hist = (i-1 + ((value - histogram_bounds[i-1]) / (v - histogram_bounds[i-1]))) / (len(histogram_bounds) - 1)
            break
    # If all buckets are covered, record a large value as the selectivity
    if hist is None:
        hist = 1 - 1e-7
    return min(max(relevantmvfsum + (1 - sum(freqs) - col_stats["null_frac"]) * hist, 1e-7), 1)


def calc_selectivity(op, value, col_stats):
    if col_stats is None:
        return -1
    if op == '=':
        return eq_selectivity(value, col_stats)
    elif op == '<':
        return lt_selectivity(value, col_stats)
    elif op == '<=':
        return min(lt_selectivity(value, col_stats) + eq_selectivity(value, col_stats), 1)
    elif op == '>':
        return 1 - min((lt_selectivity(value, col_stats) + eq_selectivity(value, col_stats)), 1)
    elif op == '>=':
        return 1 - lt_selectivity(value, col_stats)
    elif op == '<>':
        return 1 - eq_selectivity(value, col_stats)
    return 1


def extract_val(val_str, col_stats):
    val = val_str.split("::")[0]
    if col_stats["data_type"] not in {'bigint', 'integer', 'numeric'}:
        # Remove single quotes, e.g., 'abc' -> abc
        return str(val[1:-1])
    if "'" in val:
        val = val[1:-1]
    return float(val)


def compute_column_selectivity(tbl_col, meta_cond, db_stats):
    '''
    :param tbl_col: date_dim.d_year
    :param meta_cond: d_year = 2000
    :param db_stats:
    :return: float
    '''
    tokens = meta_cond.split(" ")
    # Check if selectivity calculation is supported
    if len(tokens) < 3 or tokens[2] in {"ALL", "ANY"} or tokens[1] not in {'=', '<', '<=', '>', '>=', '<>'} or tokens[2].startswith("$"):
        raise Exception(f"Selectivity calculation not supported for this condition, col: {tbl_col} cond: {meta_cond}")

    col = tokens[0]
    if col not in tbl_col:
        raise Exception(f"col: {tbl_col} column not in the condition: {meta_cond}")

    if tbl_col not in db_stats:
        raise Exception(f"col: {tbl_col} column lacks statistical information")

    op = tokens[1]
    col_stats = db_stats[tbl_col]
    # Value parsing
    val = extract_val(" ".join(tokens[2:]), col_stats)
    selectivity = calc_selectivity(op, val, col_stats)
    return selectivity


if __name__ == "__main__":
    import json
    with open("../stats_data/indexselection_tpcds___10_stats.json") as f:
        stats = json.load(f)
    print(compute_column_selectivity("date_dim.d_date", "d_date >= '2001-01-12'::date", stats))
