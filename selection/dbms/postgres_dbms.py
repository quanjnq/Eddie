import logging
import re

import psycopg2

from selection.database_connector import DatabaseConnector


class PostgresDatabaseConnector(DatabaseConnector):
    def __init__(self, db_name, autocommit=False, host=None, port=None, user=None, password=None):
        DatabaseConnector.__init__(self, db_name, autocommit=autocommit)
        self.db_system = "postgres"
        self._connection = None
        self.db_name = db_name
        if not self.db_name:
            self.db_name = "postgres"

        self.host = host
        self.port = port
        self.user = user
        self.password = password

        self.create_connection()
        self.set_random_seed()

        logging.debug("Postgres connector created: {}".format(db_name))

    def create_connection(self):
        if self._connection:
            self.close()
        self._connection = psycopg2.connect(host=self.host,
                                        database=self.db_name,
                                        port=self.port,
                                        user=self.user,
                                        password=self.password)
        self._connection.autocommit = self.autocommit
        self._cursor = self._connection.cursor()

    def enable_simulation(self):
        self.exec_only("create extension hypopg")
        self.commit()

    def database_names(self):
        result = self.exec_fetch("select datname from pg_database", False)
        return [x[0] for x in result]

    # Updates query syntax to work in PostgreSQL
    def update_query_text(self, text):
        text = text.replace(";\nlimit ", " limit ").replace("limit -1", "")
        text = re.sub(r" ([0-9]+) days\)", r" interval '\1 days')", text)
        text = self._add_alias_subquery(text)
        return text

    # PostgreSQL requires an alias for subqueries
    def _add_alias_subquery(self, query_text):
        text = query_text.lower()
        positions = []
        for match in re.finditer(r"((from)|,)[  \n]*\(", text):
            counter = 1
            pos = match.span()[1]
            while counter > 0:
                char = text[pos]
                if char == "(":
                    counter += 1
                elif char == ")":
                    counter -= 1
                pos += 1
            next_word = query_text[pos:].lstrip().split(" ")[0].split("\n")[0]
            if next_word[0] in [")", ","] or next_word in [
                "limit",
                "group",
                "order",
                "where",
            ]:
                positions.append(pos)
        for pos in sorted(positions, reverse=True):
            query_text = query_text[:pos] + " as alias123 " + query_text[pos:]
        return query_text

    def create_database(self, database_name):
        self.exec_only("create database {}".format(database_name))
        logging.info("Database {} created".format(database_name))

    def import_data(self, table, path, delimiter="|"):
        with open(path, "r") as file:
            self._cursor.copy_from(file, table, sep=delimiter, null="")

    def indexes_size(self):
        # Returns size in bytes
        statement = (
            "select sum(pg_indexes_size(table_name::text)) from "
            "(select table_name from information_schema.tables "
            "where table_schema='public') as all_tables"
        )
        result = self.exec_fetch(statement)
        return result[0]

    def drop_database(self, database_name):
        statement = f"DROP DATABASE {database_name};"
        self.exec_only(statement)

        logging.info(f"Database {database_name} dropped")

    def create_statistics(self):
        logging.info("Postgres: Run `analyze`")
        self.commit()
        self._connection.autocommit = True
        self.exec_only("analyze")
        self._connection.autocommit = self.autocommit

    def set_random_seed(self, value=0.17):
        logging.info(f"Postgres: Set random seed `SELECT setseed({value})`")
        self.exec_only(f"SELECT setseed({value})")

    def supports_index_simulation(self):
        if self.db_system == "postgres":
            return True
        return False

    def _simulate_index(self, index):
        table_name = index.table()
        cols = index.joined_column_names().split(",")
        cols = [f'"{col}"' for col in cols]
        joined_column_names_str = ",".join(cols)
        statement = (
            "select * from hypopg_create_index( "
            "'create index on  " + f'"{table_name}"'
            f"({joined_column_names_str})')"
        )
        result = self.exec_fetch(statement)
        return result

    def _drop_simulated_index(self, oid):
        statement = f"select * from hypopg_drop_index({oid})"
        result = self.exec_fetch(statement)

        assert result[0] is True, f"Could not drop simulated index with oid = {oid}."

    def create_index(self, index):
        table_name = index.table()
        statement = (
            f"create index {index.index_idx()} "
            f"on {table_name} ({index.joined_column_names()})"
        )
        self.exec_only(statement)
        size = self.exec_fetch(
            f"select relpages from pg_class c " f"where c.relname = '{index.index_idx()}'"
        )
        size = size[0]
        index.estimated_size = size * 8 * 1024

    def drop_indexes(self):
        # logging.info("Dropping indexes")
        stmt = "select indexname from pg_indexes where schemaname='public'"
        indexes = self.exec_fetch(stmt, one=False)
        for index in indexes:
            index_name = index[0]
            if "pkey" in index_name:
                # logging.info(f"Dropping indexes pass drop pkey {index_name}")
                drop_stmt = f'alter table "{index_name.split("_pkey")[0]}" drop constraint "{index_name}";'
            else:
                drop_stmt = 'drop index "{}"'.format(index_name)
            logging.debug("Dropping index {}".format(index_name))
            self.exec_only(drop_stmt)

    # PostgreSQL expects the timeout in milliseconds
    def exec_query(self, query, timeout=None, cost_evaluation=False, print_err=True):
        # Committing to not lose indexes after timeout
        if not cost_evaluation:
            self._connection.commit()
        query_text = self._prepare_query(query)
        if timeout:
            set_timeout = f"set statement_timeout={timeout}"
            self.exec_only(set_timeout)
        statement = f"explain (verbose, analyze, buffers, format json) {query_text}"
        try:
            plan = self.exec_fetch(statement, one=True)[0][0]["Plan"]
            result = plan["Actual Total Time"], plan
        except Exception as e:
            # logging.error(f"{query.nr} {query.text}, timeout {timeout} {e}")
            if print_err:
                logging.error(f"{query.nr} , timeout {timeout}")
            # self._connection.rollback()
            result = None, self._get_plan(query)
        # Disable timeout
        self._cursor.execute("set statement_timeout = 0")
        self._cleanup_query(query)
        return result

    # PostgreSQL expects the timeout in milliseconds
    def exec_explain_query(self, query, timeout=None, cost_evaluation=False, use_txt=False):
        # Committing to not lose indexes after timeout
        if not cost_evaluation:
            self._connection.commit()
        query_text = self._prepare_query(query, use_txt=use_txt)
        if timeout:
            set_timeout = f"set statement_timeout={timeout}"
            self.exec_only(set_timeout)
        statement = f"explain (verbose, format json) {query_text}"
        try:
            plan = self.exec_fetch(statement, one=True)[0][0]["Plan"]
            result = plan["Total Cost"], plan
        except Exception as e:
            logging.error(f"{query.nr}, {e}")
            self._connection.rollback()
            result = None, self._get_plan(query)
        # Disable timeout
        self._cursor.execute("set statement_timeout = 0")
        self._cleanup_query(query)
        return result

    def exec_fetchall(self, query):
        self._cursor.execute(query)
        return self._cursor.fetchall()

    def _cleanup_query(self, query):
        for query_statement in query.text.split(";"):
            try:
                if "drop view" in query_statement:
                    self.exec_only(query_statement)
                    self.commit()
            except Exception as e:
                logging.error(f"drop view Exception: {query.nr} {query_statement} {e}")

    def _get_cost(self, query):
        query_plan = self._get_plan(query)
        total_cost = query_plan["Total Cost"]
        return total_cost

    def _get_plan(self, query):
        query_text = self._prepare_query(query)
        statement = f"explain (verbose, format json) {query_text}"
        query_plan = self.exec_fetch(statement)[0][0]["Plan"]
        self._cleanup_query(query)
        return query_plan

    def number_of_indexes(self):
        statement = """select count(*) from pg_indexes
                       where schemaname = 'public'"""
        result = self.exec_fetch(statement)
        return result[0]

    def table_exists(self, table_name):
        statement = f"""SELECT EXISTS (
            SELECT 1
            FROM pg_tables
            WHERE tablename = '{table_name}');"""
        result = self.exec_fetch(statement)
        return result[0]

    def database_exists(self, database_name):
        statement = f"""SELECT EXISTS (
            SELECT 1
            FROM pg_database
            WHERE datname = '{database_name}');"""
        result = self.exec_fetch(statement)
        return result[0]

    def get_ind_plan_cost(self, query, indexes, mode="hypo"):
        self.create_indexes(indexes, mode)

        stmt = f"explain (verbose, format json) {query}"
        query_plan = self.exec_fetch(stmt)[0][0]["Plan"]
        # drop view
        # self._cleanup_query(query)
        total_cost = query_plan["Total Cost"]

        if mode == "hypo":
            self.drop_hypo_indexes()
        else:
            self.drop_indexes()

        return query_plan, total_cost

    def drop_hypo_indexes(self):
        logging.info("Dropping hypo indexes")
        stmt = "SELECT * FROM hypopg_reset();"
        self.exec_only(stmt)

    def create_indexes(self, indexes, mode="hypo"):
        """
        :param mode: 'hypo' or not
        :param indexes: table#col1,col2#col1,col2,col3
        :return:
        """
        try:
            for index in indexes:
                index_def = index.split("#")
                index_name = index.replace("#", "_").replace(",", "_")
                stmt = f"create index on {index_def[0]} ({index_def[1]})"
                if len(index_def) == 3:
                    stmt += f" include ({index_def[2]})"
                if mode == "hypo":
                    stmt = f"select * from hypopg_create_index('{stmt}')"
                self.exec_only(stmt)
        except Exception as e:
            print(e)
            print(stmt)

    def hypopg_reset(self):
        logging.info("Dropping hypo indexes")
        # print("Dropping hypo indexes")
        stmt = "SELECT * FROM hypopg_reset();"
        self.exec_only(stmt)
        self.simulated_indexes = 0
        self.cost_estimations = 0
        self.cost_estimation_duration = 0
        self.index_simulation_duration = 0