import random
from workload_gen.wheregen import *
from workload_gen.randoms import random_int_range, random_int_minmax
from workload_gen.common import get_selectable_column
from workload_gen.wheregen import where_generator
from sqlalchemy import *
from workload_gen import conf

class Scope(object):
    # to model column/table reference visibility
    # store a list of subquery named as "d_int"
    # when the size exceeds a given limit, randomly remove one
    table_ref_list = []
    # this is for scalar subquery construction
    table_ref_stmt_list = []

    def __init__(self,
                 columnlist=None,
                 tablelist=None,
                 parent=None,
                 table_spec=None):
        # available for column and table ref
        if (columnlist is not None):
            self.column_list = columnlist
        else:
            self.column_list = []
        if (tablelist is not None):
            self.table_list = tablelist
        else:
            self.table_list = []
        if parent is not None:
            self.column_list = parent.column_list
            self.table_list = parent.table_list
            self.stmt_seq = parent.stmt_seq
        self.alc_tables = None
        # Counters for prefixed stmt-unique identifiers
        # shared_ptr<map<string,unsigned int> >
        # "ref_" to an index
        self.stmt_seq = {}

    def add_alc(self, alc_tables):
        self.alc_tables = alc_tables


class Prod(object):
    def __init__(self, name, spec, spec_stat, scope, parent=None):
        self.pprod = parent
        self.name = name
        # record table's column and its type
        self.spec = spec
        # record tables' simple stat
        self.spec_stat = spec_stat
        # model column/table reference visibility
        self.scope = scope


class From_Clause(Prod):
    # static variable for the class, store a list of subquery from previous runs
    def __init__(self, name, spec, spec_stat, scope, subqueries, parent=None, prob_conf=None):
        super().__init__(name, spec, spec_stat, scope, parent)
        self.prob_conf = prob_conf
        self.subqueries = subqueries

    def get_random_from_clause(self, fuzzer=None, query_spec=None, recur_level=0):
        # branch choice either simple table, subquery, or new joined stmt
        if recur_level > 3:
            return random.choice(self.scope.alc_tables), None, None
        randval = random_int_range(1000)
        if randval < 100: 
            stmt = fuzzer.gen(query_spec, recur_level)
            return stmt, None, None
        else:
            join_prob = random_int_range(1000)
            if join_prob < 100:
                return self.get_joined_table(fuzzer)
            else:
                return random.choice(self.scope.alc_tables), None, None
            
    def get_joined_table(self, fuzzer):
        # step 1. get two table_ref
        # step 2. perform join operation on two table_ref
        
        table_a_name = random.choice(list(fuzzer.joinkey.keys()))
        table_b_name = random.choice(list(fuzzer.joinkey[table_a_name]))
        
        col_a_b_name =  fuzzer.joinkey[table_a_name][table_b_name][0]
        col_a_name = col_a_b_name[0]
        col_b_name = col_a_b_name[1]

        table_a = fuzzer.table_name_2_alc_table[table_a_name]
        table_b = fuzzer.table_name_2_alc_table[table_b_name]
        col_a = fuzzer.table_name_2_alc_columns[table_a_name][col_a_name]
        col_b = fuzzer.table_name_2_alc_columns[table_b_name][col_b_name]
        
        join_type = random.choice(["inner", "full", "left", "right"])
        
        if join_type == "left":
            j = table_a.join(table_b, col_a.__eq__(col_b), isouter=True)
        elif join_type == "right":
            j = table_b.join(table_a, col_a.__eq__(col_b), isouter=True)
        elif join_type == "inner":
            j = table_a.join(table_b, col_a.__eq__(col_b), isouter=False)
        elif join_type == "full":
            j = table_a.outerjoin(table_b, col_a.__eq__(col_b), full=True)
        
        return table_a, table_b, j


class Select_List(Prod):
    def __init__(self, name, spec, spec_stat, scope, parent=None):
        super().__init__(name, spec, spec_stat, scope, parent)
        self.value_exprs = []
        # columns is for subquery constrction, renaming purpose
        self.columns = 0
        # derived_table is for gen_select_statement and subquery construction
        self.derived_table = {}

    def gen_select_expr(self, selectable_columns, groupby_columns, number_columns=None, recur_level=0, max_selectable_column_num=3):
        selectable_columns_length = len(selectable_columns)
        if selectable_columns_length == 0:
            return None, None
        if (number_columns is None):
            number_columns = random_int_range(min(selectable_columns_length, max_selectable_column_num))
        chosen_columns_obj = random.sample(selectable_columns, number_columns)
        out = chosen_columns_obj

        use_win_func = False
        winable_columns = selectable_columns
        if len(groupby_columns) > 0:
            winable_columns = groupby_columns
            use_win_func = True
        elif (random_int_range(100) > 95):
            use_win_func = True
        
        # functions to increase the variety of selectable objects
        if use_win_func:
            for i in range(len(out)):
                if out[i] in groupby_columns:
                    continue
                added_distinct = False
                if (random_int_range(10) > 5):
                    new_type = out[i].type
                    label_name = "c_" + str(recur_level) + str(i)
                    out[i] = type_coerce((func.distinct(out[i])), new_type).label(name=label_name)
                    added_distinct = True
                    
                # second generate function
                selectable_func_list = get_compatible_function(out[i])
                selected_func = random.choice(selectable_func_list)
                new_type = out[i].type
                # only count function would change the type of the column
                if selected_func._FunctionGenerator__names == func.count._FunctionGenerator__names:
                    new_type = Float
                else:
                    label_name = "c_" + str(recur_level) + str(i)
                    out[i] = type_coerce(
                        (selected_func(out[i])),
                        new_type).label(name=label_name)
        self.value_exprs = out
        return self.value_exprs, use_win_func


class QuerySpec(Prod):
    # top class for generating select statement
    def __init__(self, name, spec, spec_stat, scope, parent=None):
        super().__init__(name, spec, spec_stat, scope, parent)
        self.from_clause = []
        self.select_list = []
        self.limit_clause = None
        self.offset_clause = None
        self.scope = scope
        self.entity_list = []
        self.subqueries = []
        # print("running constructor for query_spec")

    def get_table_idx_from_column_name(self, column_name):
        # input: convoluted column name resulting from alias rename
        # output: table_idx and correspond simple columname
        suffix_column_name = column_name.split(".")[-1]
        for i in range(len(self.spec_stat)):
            t_spec = self.spec_stat[i]
            for c in t_spec.column_name:
                if c == suffix_column_name:
                    return i, suffix_column_name
        return None, None

    def gen_statement(self,
                             fuzzer=None,
                             select_column_number=None,
                             force_simple_from=False, 
                             min_groupby_cols=1,
                             max_groupby_cols=3,
                             recur_level=0):
        # parameter needed: prod, scope
        # 1. ########## generate from_clause ##########
        #     get a random table
        base_table = False
        
        from_ins = From_Clause(self.name, self.spec, self.spec_stat,
                               self.scope, self.subqueries)
        
        from_clause1, from_clause2, joined_from = from_ins.get_random_from_clause(
            fuzzer=fuzzer, query_spec=self, recur_level=recur_level)
        if from_clause1 is None:
            return (None, None), None, None, None, None, None, None, None, None
        
        if ("Table" in str(type(from_clause1))):
            base_table = True
        
        selectable_columns = []
        if joined_from is not None:
            selectable_columns = get_selectable_column(from_clause1, fuzzer.col_backlist) \
                                + get_selectable_column(from_clause2, fuzzer.col_backlist)
        else:
            selectable_columns = get_selectable_column(from_clause1, fuzzer.col_backlist)
            
        real_selectable_columns = []
        db_stats = fuzzer.db_stats
        for random_column_object in selectable_columns:
            columnname = str(random_column_object).split(".")[-1]
            tbl_col = str(random_column_object)
            if tbl_col not in db_stats:
                real_selectable_columns.append(random_column_object)
                continue
                
            
            if db_stats[tbl_col]["data_type"] in {'bigint', 'integer', 'numeric'}:
                real_selectable_columns.append(random_column_object)
            
        selectable_columns = real_selectable_columns
        if len(selectable_columns) <= 1:
            return (None, None), None, None, None, None, None, None, None, None
        # 2. ########## generate group columns ##########
        where_clause = None
        expected_num_where = random.choice([3,3,3,1,2]) # Generate index-related queries whenever possible
        num_where = min(expected_num_where, random_int_range(len(selectable_columns)))
        if num_where != expected_num_where:
            return (None, None), None, None, None, None, None, None, None, None
        where_cols = random.sample(selectable_columns, num_where)
        if selectable_columns is not None and (random_int_range(1000) < fuzzer.prob_config["where"]):
            where_clause = self.gen_where_clause(where_cols, fuzzer)
        
        
        is_select_distinct = False
        selectable_groupby_columns = where_cols
        groupby_columns = []
        if  (random_int_range(1000) < fuzzer.prob_config["group"]):
            if not selectable_groupby_columns or len(selectable_groupby_columns) < 1:
                return (None, None), None, None, None, None, None, None, None, None
            min_k = min(min_groupby_cols, len(selectable_groupby_columns))
            max_k = min(max_groupby_cols, len(selectable_groupby_columns))
            
            expected_num_cols = random.choice([2,2,2,1,3]) # Generate index-related queries whenever possible
            num_cols = min(expected_num_cols, len(selectable_groupby_columns))
            if expected_num_cols != num_cols:
                return (None, None), None, None, None, None, None, None, None, None
             
            groupby_columns = random.sample(
                selectable_groupby_columns,
                num_cols)
        # distinct entire select
        elif (random_int_range(1000) < fuzzer.prob_config["distinct"]):
            is_select_distinct = True
            
        # ########## should decide where to select from by this point ##########
        # 3. ########## generate select_expr ##########
        select_list = Select_List(self.name, self.spec, self.spec_stat,
                                  self.scope)
        
        select_list_expr, use_func = select_list.gen_select_expr(
            selectable_columns, groupby_columns, number_columns=select_column_number, recur_level=recur_level, max_selectable_column_num=fuzzer.max_selectable_column_num)

        return (select_list_expr, use_func), where_clause, selectable_columns, joined_from, base_table, groupby_columns, is_select_distinct, where_cols, recur_level

    def gen_where_clause(self, selectable_columns_obj, fuzzer):
        # generate where clause according to the sqlalchemy column
        # 1) select a sql alchemy column
        
        where_clause_list = []
        sel_li = None
        
        for i, random_column_object in enumerate(selectable_columns_obj):
            # get the column object
            # handle joined table where column names do not always
            # belong to the same table
            columnname = str(random_column_object).split(".")[-1]
            table_idx, columnname = self.get_table_idx_from_column_name(
                columnname)
            if table_idx is None:
                # this is not an original column
                column_where = where_generator(random_column_object, None,
                                                None, None, None, fuzzer)
            else:
                column_stat = self.spec_stat[table_idx].ret_stat(
                    columnname)
                column_data = self.spec_stat[table_idx].ret_string(
                    columnname)
                column_data = [d for d in column_data if d]
                column_where = where_generator(random_column_object, None,
                                                column_stat, None,
                                                column_data, fuzzer, sel_li[i] if sel_li else None)
            if column_where is not None:
                where_clause_list.append(column_where)

            random_idx_list = list(range(len(Scope.table_ref_stmt_list)))
            random.shuffle(random_idx_list)

            # choose from an existing stmt
            for idx in random_idx_list:
                s = Scope.table_ref_stmt_list[idx]
                if len(s.columns) == 1:
                    srctype = str(get_selectable_column(s)[0].type)
                    column_stat = None
                    column_data = None
                    if "CHAR" in srctype:
                        column_data = [conf.SCALAR_STR]
                    elif "FLOAT" in srctype or "INT" in srctype or "NUMERIC" in srctype:
                        column_data = [int(conf.SCALAR_INT)]
                        column_stat = [int(conf.SCALAR_INT)]
                    else:
                        continue
                    scalar_stmt = s.limit(1).as_scalar()
                    column_where = where_generator(scalar_stmt, None,
                                                   column_stat, None,
                                                   column_data)
                    if column_where is not None:
                        where_clause_list.append(column_where)
                        break

        # begin merging the where clause
        parenthesis = False
        while (len(where_clause_list) > 1):
            where1 = where_clause_list[0]
            if where1 is None:
                where_clause_list.remove(where1)
                continue
            where2 = where_clause_list[1]
            if where2 is None:
                where_clause_list.remove(where2)
                continue
            combined_where = None
            if (random_int_range(1000) <= fuzzer.prob_config["logical_or"]):
                logical_op = "or"
            else:
                logical_op = "and"
            if parenthesis is False:
                combined_where = combine_condition(where1, where2, logical_op)
                parenthesis = True
            else:
                combined_where = combine_parenthesis(where1, where2, logical_op)
            where_clause_list.remove(where1)
            where_clause_list.remove(where2)
            where_clause_list.insert(0, combined_where)

        if len(where_clause_list) > 0:
            return where_clause_list[0]
        else:
            return None
