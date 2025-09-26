
import numpy as np

node_types2idx = {'Unique':0,'Hash Join':1,'Bitmap Heap Scan':2,'Materialize':3,'SetOp':4,'Subquery Scan':5,'Aggregate':6,'BitmapAnd':7,'Gather Merge':8,'WindowAgg':9,'Sort':10,'Gather':11,'Index Scan':12,'Merge Join':13,'Bitmap Index Scan':14,'Nested Loop':15,'Index Only Scan':16,'CTE Scan':17,'Hash':18,'BitmapOr':19,'Limit':20,'Result':21,'Merge Append':22,'Append':23,'Group':24,'Seq Scan':25}


CHANNELS = ['EstNodeCost', 'EstRows', 'EstBytes', 'RowsWeightedSum', 'BytesWeightedSum']
NUM_CHANNELS = len(CHANNELS)
NUM_NODE_TYPES = len(node_types2idx)


def aggregate_by_channel_and_node_type(results):
    
    aggregated_data = {}
    for channel in CHANNELS:
        aggregated_data[channel] = {}
        for node_type_idx in range(NUM_NODE_TYPES):
            aggregated_data[channel][node_type_idx] = 0.0
    
    for stats in results:
        node_type = stats['Node Type']
        node_type_idx = node_types2idx.get(node_type, -1)
        
        if node_type_idx == -1:
            print(f"pass node {node_type}")
            continue
        
        aggregated_data['EstNodeCost'][node_type_idx] += stats['EstNodeCost']
        aggregated_data['EstRows'][node_type_idx] += stats['EstRows']
        aggregated_data['EstBytes'][node_type_idx] += stats['EstBytes']
        aggregated_data['RowsWeightedSum'][node_type_idx] += stats['LeafWeightEst-RowsWeightedSum']
        aggregated_data['BytesWeightedSum'][node_type_idx] += stats['LeafWeightEst-BytesWeightedSum']
    
    matrix = []
    for channel in CHANNELS:
        row = []
        for node_type_idx in range(NUM_NODE_TYPES):
            row.append(aggregated_data[channel][node_type_idx])
        matrix.append(row)
    
    return aggregated_data, matrix


def pair_diff(matrix1, matrix2):
    matrix1_np = np.array(matrix1)
    matrix2_np = np.array(matrix2)
    diff_matrix = matrix2_np - matrix1_np
    return diff_matrix


def pair_diff_ratio(matrix1, matrix2, clip_value=1e4):
    
    matrix1_np = np.array(matrix1)
    matrix2_np = np.array(matrix2)
    
    diff_matrix = matrix2_np - matrix1_np
    
    matrix1_clipped = np.where(np.abs(matrix1_np) < 1e-8, clip_value, matrix1_np)
    
    ratio_matrix = diff_matrix / matrix1_clipped
    
    return ratio_matrix


def pair_diff_normalized(matrix1, matrix2):
    matrix1_np = np.array(matrix1)
    matrix2_np = np.array(matrix2)
    
    diff_matrix = matrix2_np - matrix1_np
    
    channel_sums = np.sum(matrix1_np, axis=1, keepdims=True) + np.sum(matrix2_np, axis=1, keepdims=True)
    
    
    channel_sums = np.where(channel_sums == 0, 1.0, channel_sums)
    
    normalized_matrix = diff_matrix / channel_sums
    
    return normalized_matrix


def print_aggregation_results(aggregated_data, matrix):
    idx2node_type = {idx: node_type for node_type, idx in node_types2idx.items()}
    
    for channel_idx, channel in enumerate(CHANNELS):
        print(f"\n【{channel}】:")
        print("-" * 60)
        for node_type_idx in range(NUM_NODE_TYPES):
            value = aggregated_data[channel][node_type_idx]
            if value > 0:  
                node_type_name = idx2node_type[node_type_idx]
                print(f"  {node_type_name} (索引{node_type_idx}): {value:.2f}")
    


def analyze_execution_plan_with_aggregation(plan_root):
    
    results = analyze_execution_plan(plan_root)
    
    aggregated_data, matrix = aggregate_by_channel_and_node_type(results)
    
    return results, aggregated_data, matrix


def calculate_node_height(node):
    if 'Plans' not in node or not node['Plans']:
        return 1
    
    child_heights = [calculate_node_height(child) for child in node['Plans']]
    return min(child_heights) + 1


def calculate_basic_stats(node):
    est_node_cost = node.get('Total Cost', 0)
    est_rows = node.get('Plan Rows', 0)
    plan_width = node.get('Plan Width', 0)
    est_bytes = est_rows * plan_width
    
    return est_node_cost, est_rows, est_bytes


def calculate_weighted_sums(node):
    
    _, est_rows, est_bytes = calculate_basic_stats(node)
    
    node_height = calculate_node_height(node)
    
    rows_weighted = est_rows * node_height
    bytes_weighted = est_bytes * node_height
    
    if 'Plans' in node and node['Plans']:
        for child in node['Plans']:
            child_rows_weighted, child_bytes_weighted = calculate_weighted_sums(child)
            rows_weighted += child_rows_weighted
            bytes_weighted += child_bytes_weighted
    
    return rows_weighted, bytes_weighted


def calculate_node_statistics(node):
    est_node_cost, est_rows, est_bytes = calculate_basic_stats(node)
    
    leaf_weight_rows_sum, leaf_weight_bytes_sum = calculate_weighted_sums(node)
    
    node_type = node.get('Node Type', 'Unknown')
    node_height = calculate_node_height(node)
    
    return {
        'Node Type': node_type,
        'Height': node_height,
        'EstNodeCost': est_node_cost,
        'EstRows': est_rows,
        'EstBytes': est_bytes,
        'LeafWeightEst-RowsWeightedSum': leaf_weight_rows_sum,
        'LeafWeightEst-BytesWeightedSum': leaf_weight_bytes_sum
    }


def analyze_execution_plan(plan_root):
    results = []
    
    def traverse_and_analyze(node, depth=0):
        stats = calculate_node_statistics(node)
        stats['Depth'] = depth
        results.append(stats)
        
        if 'Plans' in node and node['Plans']:
            for child in node['Plans']:
                traverse_and_analyze(child, depth + 1)
    
    traverse_and_analyze(plan_root)
    return results


def process_plan_pair(init_plan, hypo_plan):
    _, _, matrix1 = analyze_execution_plan_with_aggregation(init_plan)
    _, _, matrix2 = analyze_execution_plan_with_aggregation(hypo_plan)
    
    pair_diff_normalized_matrix = pair_diff_normalized(matrix1, matrix2)
    
    return pair_diff_normalized_matrix


example_plan1 = {
    "Node Type": "Hash Join",
    "Total Cost": 35,
    "Plan Rows": 200,
    "Plan Width": 10,
    "Plans": [
        {
            "Node Type": "Hash Join",
            "Total Cost": 20,
            "Plan Rows": 200,
            "Plan Width": 10,
            "Plans": [
                {
                    "Node Type": "Index Scan",
                    "Total Cost": 10,
                    "Plan Rows": 200,
                    "Plan Width": 10,
                    "Plans": []
                },
                {
                    "Node Type": "Seq Scan",
                    "Total Cost": 30,
                    "Plan Rows": 1000,
                    "Plan Width": 10,
                    "Plans": []
                }
            ]
        },
        {
            "Node Type": "Seq Scan",
            "Total Cost": 50,
            "Plan Rows": 1000,
            "Plan Width": 10,
            "Plans": []
        }
    ]
}

example_plan2 = {
    "Node Type": "Hash Join",
    "Total Cost": 35,
    "Plan Rows": 200,
    "Plan Width": 10,
    "Plans": [
        {
            "Node Type": "Hash Join",
            "Total Cost": 20,
            "Plan Rows": 1000,
            "Plan Width": 10,
            "Plans": [
                {
                    "Node Type": "Index Scan",
                    "Total Cost": 20,
                    "Plan Rows": 1000,
                    "Plan Width": 10,
                    "Plans": []
                },
                {
                    "Node Type": "Seq Scan",
                    "Total Cost": 30,
                    "Plan Rows": 1000,
                    "Plan Width": 10,
                    "Plans": []
                }
            ]
        },
        {
            "Node Type": "Index Scan",
            "Total Cost": 10,
            "Plan Rows": 200,
            "Plan Width": 10,
            "Plans": []
        }
    ]
}


if __name__ == "__main__":
    results1, aggregated_data1, matrix1 = analyze_execution_plan_with_aggregation(example_plan1)
    
    for i, stats in enumerate(results1):
        print(f"  EstNodeCost: {stats['EstNodeCost']:.2f}")
        print(f"  EstRows: {stats['EstRows']}")
        print(f"  EstBytes: {stats['EstBytes']}")
        print(f"  LeafWeightEst-RowsWeightedSum: {stats['LeafWeightEst-RowsWeightedSum']}")
        print(f"  LeafWeightEst-BytesWeightedSum: {stats['LeafWeightEst-BytesWeightedSum']}")
        print("-" * 40)
    
    
    print_aggregation_results(aggregated_data1, matrix1)
    
    results2, aggregated_data2, matrix2 = analyze_execution_plan_with_aggregation(example_plan2)
    
    for i, stats in enumerate(results2):
        print(f"  EstNodeCost: {stats['EstNodeCost']:.2f}")
        print(f"  EstRows: {stats['EstRows']}")
        print(f"  EstBytes: {stats['EstBytes']}")
        print(f"  LeafWeightEst-RowsWeightedSum: {stats['LeafWeightEst-RowsWeightedSum']}")
        print(f"  LeafWeightEst-BytesWeightedSum: {stats['LeafWeightEst-BytesWeightedSum']}")
        print("-" * 40)
    

    print_aggregation_results(aggregated_data2, matrix2)
    
    matrix1_np = np.array(matrix1)
    matrix2_np = np.array(matrix2)
    
    print("Matrix1 (P1 - example_plan1):")
    print(matrix1_np)
    print("\nMatrix2 (P2 - example_plan2):")
    print(matrix2_np)
    
    
    print("-" * 40)
    diff_result = pair_diff(matrix1, matrix2)
    print(diff_result)
    
    
    print("-" * 40)
    ratio_result = pair_diff_ratio(matrix1, matrix2)
    print(ratio_result)
    
    print("-" * 40)
    normalized_result = pair_diff_normalized(matrix1, matrix2)
    print(normalized_result)
    
    channel_sums_p1 = np.sum(matrix1_np, axis=1)
    channel_sums_p2 = np.sum(matrix2_np, axis=1)
    total_channel_sums = channel_sums_p1 + channel_sums_p2
    for i, (channel, sum_p1, sum_p2, sum_total) in enumerate(zip(CHANNELS, channel_sums_p1, channel_sums_p2, total_channel_sums)):
        print(f"  {channel}: P1={sum_p1:.2f} + P2={sum_p2:.2f} = {sum_total:.2f}")

