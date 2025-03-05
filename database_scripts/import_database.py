import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import random
from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from selection.table_generator import TableGenerator
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


def import_database(workload_name, scale_factor, explicit_database_name=None):
    logging.info(f"Start importing the database, workload_name:{workload_name}, scale_factor:{scale_factor}")
    random.seed(0)
    generating_connector = PostgresDatabaseConnector(None, autocommit=True)
    table_generator = TableGenerator(
        workload_name, scale_factor, generating_connector, explicit_database_name
    )
    database_name = table_generator.database_name()
    logging.info(f"Database {database_name} is successfully imported")



if __name__ == "__main__":
    import_configs = [{"workload_name": "tpcds", "scale_factor_list": [10, 5]}, {"workload_name": "tpch", "scale_factor_list": [10, 5]}]
    for cfg in import_configs:
        workload_name = cfg["workload_name"]
        for scale_factor in cfg["scale_factor_list"]:
            import_database(workload_name, scale_factor)
