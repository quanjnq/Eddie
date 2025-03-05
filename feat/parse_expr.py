import re
# JOIN expression recognition
JOIN_EXPR_OP_JOIN_EXPR = re.compile(r'^\((.+)\)\s(=|<>|<|>|<=|>=)\s\((.+)\)$')
JOIN_EXPR_OP_JOIN_COL = re.compile(r'^\((.+)\)\s(=|<>|<|>|<=|>=)\s([a-zA-Z_][a-zA-Z0-9_.]*)$')


class BinaryExprNode:
    def __init__(self, predicate=None, left=None, right=None, logical_op=None):
        self.left = left
        self.right = right
        self.logical_op = logical_op
        self.predicate = predicate

    def __str__(self):
        return self.predicate if self.logical_op is None else self.logical_op


def is_valid_parentheses(s: str) -> bool:
    stack = []
    mapping = {')': '('}

    for char in s:
        if char in mapping.values():  
            stack.append(char)
        elif char in mapping.keys(): 
            if not stack or stack.pop() != mapping[char]:  # Check if the stack is empty or matched
                return False

    return not stack


def build_binary_node(expr, deep):
    if deep > 100:
        raise Exception(f"The recursive depth exceeds the threshold: {deep}, expr:{expr}")
    if len(expr) <= 2:
        return None
    stack = []

    # Column type conversion processing, parentheses removed
    expr = expr[1:-1]
    left, right = 0, -1
    if expr[0] == '(':
        for i in range(1, len(expr)):
            c = expr[i]
            if c == ')':
                right = i
                break
            if c == '(' or c == ' ':
                break
    expr_li = list(expr)
    if right > left:
        expr_li.pop(right)
        expr_li.pop(left)
    expr = "".join(expr_li)

    # Child nodes
    if expr[0] != '(':
        return BinaryExprNode(predicate=expr)
    if "AND" not in expr and "OR" not in expr:
        match = re.match(JOIN_EXPR_OP_JOIN_EXPR, expr) or re.match(JOIN_EXPR_OP_JOIN_COL, expr)
    else:
        match = False

    if match:
        left_val = match.group(1)  
        cmp_op = match.group(2)  
        right_val = match.group(3)
        return BinaryExprNode(predicate=f"{left_val} {cmp_op} {right_val}")

    sub_expr = []
    pos = 0
    lg_op = None
    while pos < len(expr):
        ch = expr[pos]
        sub_expr.append(ch)
        if ch == ')':
            stack.pop()
        elif ch == '(':
            stack.append(ch)

        if len(stack) == 0 :
            op = []
            pos += 1
            while pos < len(expr):
                if expr[pos] != '(':
                    op.append(expr[pos])
                else:
                    break
                pos += 1
            if len(op) > 0:
                lg_op = "".join(op).strip()
            break
        if len(sub_expr) > 0:
            pos += 1
    left = build_binary_node("".join(sub_expr), deep + 1)
    rexpr = expr[pos:] if is_valid_parentheses(expr[pos+1:-1]) else '('+expr[pos:]+')'
    right = build_binary_node(rexpr, deep+1)
    return BinaryExprNode(left=left, logical_op=lg_op, right=right)

def print_binary_tree(node, level=0):
    if node is not None:
        print(' ' * 4 * level + '->', node)
        if node.left is not None:
            print_binary_tree(node.left, level+1)
        if node.right is not None:
            print_binary_tree(node.right, level+1)


if __name__ == "__main__":
    # Test cases
    express = "(((i_category = ANY ('{Jewelry,Sports,Books}'::bpchar[]) OR ((d_date >= '2001-01-12'::date) AND (d_date <= '2001-02-11 00:00:00'::timestamp without time zone))) ) AND (((cd_gender = 'M'::bpchar) AND (cd_marital_status = 'W'::bpchar) AND (cd_education_status = 'College'::bpchar)) OR ((store_sales.ss_item_sk = catalog_sales.cs_item_sk) AND (store_sales.ss_customer_sk = catalog_sales.cs_bill_customer_sk) OR ((a = 0) AND (b = 0)))))"
    express = "((year_total > '0'::numeric) AND (sale_type = 's'::text) AND (dyear = 2001))"
    express = "((c_mktsegment): = ANY ('{a,b}'::bpchar[]))"
    express = "((i_manufact = i1.i_manufact) AND (((i_category = 'Women'::bpchar) AND ((i_color = 'midnight'::bpchar) OR (i_color = 'drab'::bpchar)) AND ((i_units = 'Dozen'::bpchar) OR (i_units = 'Ton'::bpchar)) AND ((i_size = 'economy'::bpchar) OR (i_size = 'small'::bpchar))) OR ((i_category = 'Women'::bpchar) AND ((i_color = 'indian'::bpchar) OR (i_color = 'mint'::bpchar)) AND ((i_units = 'Carton'::bpchar) OR (i_units = 'Dram'::bpchar)) AND ((i_size = 'petite'::bpchar) OR (i_size = 'medium'::bpchar))) OR ((i_category = 'Men'::bpchar) AND ((i_color = 'peru'::bpchar) OR (i_color = 'smoke'::bpchar)) AND ((i_units = 'Oz'::bpchar) OR (i_units = 'Tsp'::bpchar)) AND ((i_size = 'extra large'::bpchar) OR (i_size = 'N/A'::bpchar))) OR ((i_category = 'Men'::bpchar) AND ((i_color = 'deep'::bpchar) OR (i_color = 'cornsilk'::bpchar)) AND ((i_units = 'N/A'::bpchar) OR (i_units = 'Gross'::bpchar)) AND ((i_size = 'economy'::bpchar) OR (i_size = 'small'::bpchar))) OR ((i_category = 'Women'::bpchar) AND ((i_color = 'cream'::bpchar) OR (i_color = 'chiffon'::bpchar)) AND ((i_units = 'Bundle'::bpchar) OR (i_units = 'Pound'::bpchar)) AND ((i_size = 'economy'::bpchar) OR (i_size = 'small'::bpchar))) OR ((i_category = 'Women'::bpchar) AND ((i_color = 'orange'::bpchar) OR (i_color = 'puff'::bpchar)) AND ((i_units = 'Gram'::bpchar) OR (i_units = 'Box'::bpchar)) AND ((i_size = 'petite'::bpchar) OR (i_size = 'medium'::bpchar))) OR ((i_category = 'Men'::bpchar) AND ((i_color = 'peach'::bpchar) OR (i_color = 'wheat'::bpchar)) AND ((i_units = 'Bunch'::bpchar) OR (i_units = 'Cup'::bpchar)) AND ((i_size = 'extra large'::bpchar) OR (i_size = 'N/A'::bpchar))) OR ((i_category = 'Men'::bpchar) AND ((i_color = 'pink'::bpchar) OR (i_color = 'firebrick'::bpchar)) AND ((i_units = 'Pallet'::bpchar) OR (i_units = 'Ounce'::bpchar)) AND ((i_size = 'economy'::bpchar) OR (i_size = 'small'::bpchar)))))"
    express = "(((wswscs_1.d_week_seq - 53)) = date_dim.d_week_seq)"
    express = "(((sum(store_sales.ss_ext_sales_price)) >= (0.9 * (sum(catalog_sales.cs_ext_sales_price)))) AND ((sum(store_sales.ss_ext_sales_price)) <= (1.1 * (sum(catalog_sales.cs_ext_sales_price)))) AND ((sum(catalog_sales.cs_ext_sales_price)) >= (0.9 * (sum(store_sales.ss_ext_sales_price)))) AND ((sum(catalog_sales.cs_ext_sales_price)) <= (1.1 * (sum(store_sales.ss_ext_sales_price)))) AND ((sum(store_sales.ss_ext_sales_price)) >= (0.9 * (sum(web_sales.ws_ext_sales_price)))) AND ((sum(store_sales.ss_ext_sales_price)) <= (1.1 * (sum(web_sales.ws_ext_sales_price)))) AND ((sum(web_sales.ws_ext_sales_price)) >= (0.9 * (sum(store_sales.ss_ext_sales_price)))) AND ((sum(web_sales.ws_ext_sales_price)) <= (1.1 * (sum(store_sales.ss_ext_sales_price)))))"
    if is_valid_parentheses(express):
        e_root = build_binary_node(express, 0)
        print_binary_tree(e_root)
    pass

