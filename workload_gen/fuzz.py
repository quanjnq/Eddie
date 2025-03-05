import random
from sqlalchemy import create_engine, Table, MetaData, text
from sqlalchemy.orm import sessionmaker
from workload_gen.common import ret_typename_from_class
from workload_gen.model import Scope, QuerySpec
from sqlalchemy import select, asc, desc
from statistics import mean
from workload_gen.randoms import *
from workload_gen.utils import is_valid_autoadmin_query
import logging
class TableStat(object):
    # Maintain statistics for each table
    def __init__(self, tablename):
        self.tablename = tablename
        self.columns = []
        self.column_name = []
        self.column_type = []

        self.columns_stat = []
        self.table_size = 0

    """
    def add_sqlalchemy_tbl(self, tbl):
        self.sqlalchemy_tbl = tbl
    """

    def add_column(self, column_name, column_type):
        self.column_name.append(column_name)
        self.column_type.append(column_type)
        self.columns.append([])

    def add_data(self, data):
        for x in range(len(data)):
            self.columns[x].append(data[x])
            self.table_size += 1

    def ret_stat(self, columnname):
        for x in range(len(self.column_name)):
            if self.column_name[x] == columnname:
                return self.columns_stat[x]
            else:
                AssertionError("No matching column name, my mistake")

    def ret_string(self, columnname):
        for x in range(len(self.column_name)):
            if self.column_name[x] == columnname:
                return self.columns[x]
            else:
                AssertionError("No matching column name, my mistake")

    @staticmethod
    def ret_table_with_tblname(sqlalchemy_tbllist, tblname):
        for idx in range(len(sqlalchemy_tbllist)):
            name = sqlalchemy_tbllist[idx].name
            if tblname == name:
                return sqlalchemy_tbllist[idx]
        return None

    @staticmethod
    def ret_tablestat_with_tblname(tbl_stat_list, tblname):
        for idx in range(len(tbl_stat_list)):
            name = tbl_stat_list[idx].tablename
            if tblname == name:
                return tbl_stat_list[idx]
        return None

    def calculate_stat(self):

        for x in range(len(self.columns)):

            # 1) if string/text ==> store length
            if self.column_type[x] == "String":
                temp_arr = []
                for y in range(len(self.columns[x])):
                    temp_arr.append(len(self.columns[x][y]))

                _min, _max, _avg = self.stat_from_arr(temp_arr)

            # 2) if DateTime
            elif self.column_type[x] == "DateTime":
                temp_arr = []
                for y in range(len(self.columns[x])):
                    temp_arr.append(
                        int(self.columns[x][y].strftime("%Y%m%d %H:%M:%S")))

                _min, _max, _avg = self.stat_from_arr(temp_arr)

            # 3) if numetic
            else:
                _min, _max, _avg = self.stat_from_arr(self.columns[x])

            self.columns_stat.append([_min, _max, _avg])

    def calculate_stat_existing_db(self, column_data, x):
        # Call once for each column, different from previous method calculate_stat and populate data
        # 1) if string/text ==> store length
        if len(column_data) == 0:
            _min, _max, _avg = None, None, None
        elif self.column_type[x] == "String":
            temp_arr = []
            for y in range(len(column_data)):
                if not column_data[y]:
                    temp_arr.append(0)
                else:
                    temp_arr.append(len(column_data[y]))

            _min, _max, _avg = self.stat_from_arr(temp_arr)
        elif isinstance((column_data[0]), str) or self.column_type[x] == "Array":
            # get stat for a char(1) column
            temp_arr = []
            for y in range(len(column_data)):
                temp_arr.append(len(column_data[y]))
            _min, _max, _avg = self.stat_from_arr(temp_arr)

        # 2) if DateTime
        elif isinstance((column_data[0]), datetime.date):
            temp_arr = []
            for y in range(len(column_data)):
                temp_arr.append(int(column_data[y].strftime("%Y%m%d")))
                

            _min, _max, _avg = self.stat_from_arr(temp_arr)
        # 3) if numetic
        else:
            _min, _max, _avg = self.stat_from_arr(column_data)

        self.columns_stat.append([_min, _max, _avg])
        self.columns[x].extend(column_data)

    def stat_from_arr(self, array):
        array = [elm for elm in array if elm]
        if len(array) == 0:
            return 0, 0, 0
        
        _min = min(array)
        _max = max(array)
        
        if isinstance(array[0], datetime.date):
            timestamps = [date.toordinal() for date in array]
            mean_timestamp = mean(timestamps)
            _avg = datetime.date.fromordinal(int(mean_timestamp))
        elif isinstance(array[0], datetime.time):
            _avg = self.average_time(array)
        else:
            _avg = mean(array)
        return _min, _max, _avg

    def time_to_seconds(self, t):
        return t.hour * 3600 + t.minute * 60 + t.second

    def average_time(self, time_list):
        total_seconds = sum(self.time_to_seconds(t) for t in time_list)
        avg_seconds = total_seconds // len(time_list)
        avg_time = timedelta(seconds=avg_seconds)

        return datetime.time(avg_time.seconds // 3600, (avg_time.seconds // 60) % 60, avg_time.seconds % 60)


class TableSpec(object):
    def __init__(self, name):
        self.table_name = name
        self.columns = []
        self.row_data = []
        self.pk_idx = None
        self.fk_idx = -1
        self.num_tuples = -1

    def add_column(self, column_name, column_type):
        self.columns.append((column_name, column_type))

class Fuzz:
    def __init__(self, prob_conf, query_conf, join_keys, connstr, db_conn):
            
        self.prob_config = prob_conf
        self.connstr = connstr
        self.joinkey = join_keys
        
        self.col_backlist = query_conf["col_backlist"]
        self.tbl_backlist = query_conf["tbl_backlist"]
        self.min_orderby_cols = query_conf["min_orderby_cols"]
        self.max_orderby_cols = query_conf["max_orderby_cols"]
        self.max_groupby_cols = query_conf["max_groupby_cols"]
        self.min_groupby_cols = query_conf["min_groupby_cols"]
        self.max_selectable_column_num = query_conf["max_selectable_cols"]
        self.max_num_where = query_conf["where_max_predicate_num"]
        self.db_conn = db_conn
        self.db_stats = query_conf["db_stats"]
        self.__load_existing_dbschema()
        
    def __tables(self, engine):
        q = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
            """
        tablenames = []
        results = engine.execute(q).fetchall()
        for tablename in results:
            tablenames.append(tablename[0])
        return tablenames

    def __load_existing_dbschema(self):
        tables = []  # Tables spec (Tableclass), name,
        tables_stat = []  # Tables statistics (TableStat class)
        postgres_engine = create_engine(self.connstr)
        self.postgres_engine = postgres_engine
        
        table_names = self.__tables(postgres_engine)
        schemameta = MetaData(postgres_engine)
        DBSession = sessionmaker(bind=postgres_engine)
        session = DBSession()
        self.session = session
        sqlalchemy_tables = []
        table_name_2_alc_table = {}
        table_name_2_alc_columns = {}
                
        for table_name in table_names:
            # Skip irrelevant tables
            if 'hypopg_list_indexes' in table_name or "dbgen_version" in table_name or table_name in self.tbl_backlist:
                continue
            
            messages = Table(table_name,
                             schemameta,
                             autoload=True,
                             autoload_with=postgres_engine)
            sqlalchemy_tables.append(messages)
            table_name_2_alc_table[table_name] = messages
            table_name_2_alc_columns[table_name] = {}
            
            table_stat = TableStat(table_name)
            table_class = TableSpec(table_name)

            table_class.pk_idx = -1
            table_class.fk_idx = -1
            results = session.query(messages)
            sample_results = (results[:5])
            column_index = 0

            for c in messages.columns:
                column_data = [i[column_index] for i in sample_results]
                # Need to use sqlalchemy type instead of real database's type
                table_class.add_column(c.name, (c.type))
                
                typename = ret_typename_from_class(c.type)
                table_stat.add_column(c.name, typename)
                # Some type may not use for intersection calculation
                column_index += 1
                table_name_2_alc_columns[table_name][c.name] = c

            tables.append(table_class)
            tables_stat.append(table_stat)
            for c in range(len(messages.columns)):
                column_data = [i[c] for i in sample_results if i[c]]
                
                tables_stat[(
                    tables_stat).index(table_stat)].calculate_stat_existing_db(
                        column_data, c)

        self.tables = tables
        self.tables_stat = tables_stat
        self.sqlalchemy_tables = sqlalchemy_tables
        self.table_name_2_alc_table = table_name_2_alc_table
        self.table_name_2_alc_columns = table_name_2_alc_columns

    def __stmt_complex(self, stmt, groupby_columns, is_select_distinct, where_cols=None, recur_level=0, use_func=False):
        
        for column in groupby_columns:
            stmt = stmt.group_by(column)

        if is_select_distinct:
            stmt = stmt.distinct()
        
        if (random_int_range(1000) <= self.prob_config["order"]):
            available_columns = []
            if (not use_func and not is_select_distinct and recur_level == 1) and where_cols:
                available_columns = where_cols
            
            if len(available_columns) <= 1:
                return None
            
            expected_num_cols = random.choice([2,2,2,1,3]) # Generate index-related queries whenever possible
            num_cols = min(expected_num_cols, len(available_columns))
            if num_cols != expected_num_cols:
                return None
            chosen_orderby_columns = random.sample(
                available_columns,
                k=num_cols)
            
            for i, column in enumerate(chosen_orderby_columns):
                if (random_int_range(100) > 5):
                    stmt = stmt.order_by(asc(column))
                else:
                    stmt = stmt.order_by(desc(column))
            
        limit_rows = 100
        stmt = stmt.limit(limit_rows)
        return stmt

    def gen(self, spec, recur_level=0):
        (select_columns, use_func), where_clause, selectable_columns, joined_from, base_table, groupby_columns, is_select_distinct, where_cols, cur_recur_level = spec.gen_statement(
            fuzzer=self, recur_level=recur_level+1, min_groupby_cols=self.min_groupby_cols, max_groupby_cols=self.max_groupby_cols)
        if select_columns is None:
            return None
        if joined_from is not None:
            stmt = select(select_columns).select_from(
                joined_from)
        else:
            stmt = select(select_columns)
        if where_clause is not None:
            stmt = stmt.where(where_clause)
        
        stmt = self.__stmt_complex(stmt, groupby_columns, is_select_distinct, where_cols, cur_recur_level, use_func)
        return stmt

    def gen_orm_queries(self, count=20, max_cost=10000000, min_cost=0):
        scope = Scope()
        scope.add_alc(self.sqlalchemy_tables)
        spec = QuerySpec("demo", self.tables, self.tables_stat, scope)
        queries = []
        pg_engine = create_engine(self.connstr)
        
        while count:
            if count % 100 == 0:
                logging.info(f"The remaining {count} queries are being generated")
            stmt = self.gen(spec)
            if stmt is None:
                continue
            compiled = stmt.compile(compile_kwargs={"literal_binds": True})
            
            if ";" in str(compiled):
                continue
            try:
                result = pg_engine.execute(text(f"EXPLAIN (FORMAT JSON) {compiled}"))
            except:
                continue
            explain = [row[0] for row in result][0]
            cost =  explain[0]["Plan"]["Total Cost"]
            query_text = str(compiled).replace("\n", " ")
            
            if cost < max_cost and cost > min_cost:
                if not is_valid_autoadmin_query(self.db_conn, query_text): # Skip invalid queries
                    continue
                queries.append(query_text)
                count -= 1
        self.session.close_all()
        self.postgres_engine.dispose()
        pg_engine.dispose()
        return queries

