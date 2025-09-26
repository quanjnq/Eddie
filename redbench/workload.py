import pandas as pd
import ast

def extract_fast_sql(csv_file_path):
    try:

        df = pd.read_csv(csv_file_path)
        

        required_columns = ['sql', 'execution_time']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"{col}")
        

        fast_queries = df[df['execution_time'] < 30]
        

        fast_sql_list = fast_queries['sql'].tolist()
        

        if not fast_queries.empty:
            fastest_queries = fast_queries.nsmallest(5, 'execution_time')
            for i, (idx, row) in enumerate(fastest_queries.iterrows(), 1):
                print(f"{i}. {row['execution_time']:.6f}s: {row['sql'][:50]}...")
        
        return fast_sql_list
        
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f" {str(e)}")
        return []

def extract_fast_sql_with_details(csv_file_path):

    try:
        df = pd.read_csv(csv_file_path)
        

        fast_queries = df[df['execution_time'] < 30]
        detailed_list = fast_queries.to_dict('records')
        
        return detailed_list
        
    except Exception as e:
        print(f" {str(e)}")
        return []

def save_fast_sql_to_file(fast_sql_list, output_file):

    try:
        with open(output_file, 'w', encoding='utf-8') as f:

            
            for i, sql in enumerate(fast_sql_list, 1):
                f.write(sql + "\n\n")
        

        
    except Exception as e:
        print(f"{str(e)}")


if __name__ == "__main__":

    csv_file = "./pg_sql_execution_results.csv"
    

    fast_sqls = extract_fast_sql(csv_file)
    
    detailed_records = extract_fast_sql_with_details(csv_file)
    

    if fast_sqls:
        output_file = "./fast_sql_queries.sql"
        save_fast_sql_to_file(fast_sqls, output_file)


        for i, sql in enumerate(fast_sqls[:3], 1):
            print(f"{i}. {sql[:100]}..." if len(sql) > 100 else f"{i}. {sql}")
    