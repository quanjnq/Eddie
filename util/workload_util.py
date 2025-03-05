from selection.workload import Column, Query, Table, Workload

def get_tables(db_connector):
    result = db_connector.exec_fetchall(
        "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname='public';"
    )
    table_names = [row[0] for row in result]

    tables = {}

    for table_name in table_names:
        table = Table(table_name)
        result = db_connector.exec_fetchall(
            "SELECT column_name "
            + "FROM information_schema.columns "
            + "WHERE table_schema = 'public' "
            + f"AND table_name = '{table_name}';"
        )
        column_names = [row[0] for row in result]
        for column_name in column_names:
            table.add_column(Column(column_name))

        tables[table_name] = table

    return tables


def store_indexable_columns(query, tables):
    for table_name in tables:
        if table_name in query.text:
            table = tables[table_name]
            for column in table.columns:
                if column.name in query.text:
                    query.columns.append(column)


def tran_workload(sql_list, db_connector):
    # Retrieve schema to search for indexable columns
    tables = get_tables(db_connector)

    queries = []
    if type(sql_list[0]) is not str:
        for nr, query_text in sql_list:
            query = Query(nr, query_text)
            db_connector.exec_explain_query(query)
            store_indexable_columns(query, tables)
            queries.append(query) 
    else:
        for query_id, query_text in enumerate(sql_list):
            query = Query(query_id+1, query_text)
            store_indexable_columns(query, tables)
            queries.append(query)

    return Workload(queries)
