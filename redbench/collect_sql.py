import os
import pandas as pd
from pathlib import Path
from functools import lru_cache

class SQLFileReader:
    def __init__(self, base_directory="~/redbench/redbench/"):
        self.base_dir = Path(base_directory).expanduser().absolute()
        self.file_cache = {}  
    
    @lru_cache(maxsize=1000) 
    def read_sql_file_cached(self, relative_path):

        full_path = self.base_dir / relative_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"FileNotFoundError: {full_path}")
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except UnicodeDecodeError:
            try:
                with open(full_path, 'r', encoding='latin-1') as f:
                    return f.read().strip()
            except:
                raise UnicodeDecodeError(f"UnicodeDecodeError: {relative_path}")
        except Exception as e:
            raise Exception(f"{relative_path} - {str(e)}")
    
    def process_variability_csv_files(self, root_dir):
        root_path = Path(root_dir).expanduser().absolute()
        all_contents = []
        processed_files = set() 

        for csv_file in root_path.rglob('*variability.csv'):
            try:

                df = pd.read_csv(csv_file)
                

                if 'filepath' not in df.columns:
                    continue
                
                
                total_rows = len(df)
                unique_files = df['filepath'].nunique()
                
                
                for index, row in df.iterrows():
                    filepath = row['filepath']
                    
                    try:
                        if filepath not in self.file_cache:
                            content = self.read_sql_file_cached(filepath)
                            self.file_cache[filepath] = content
                            
                        
                        all_contents.append({
                            'source_csv': str(csv_file),
                            'filepath': str(filepath),
                            'content': self.file_cache[filepath],
                            'num_joins_user': row.get('num_joins_in_user_query', None),
                            'num_joins_benchmark': row.get('num_joins_in_benchmark_query', None),
                            'query_id': row.get('query_id', None)
                        })
                        
                    except FileNotFoundError:
                        print(f"FileNotFoundError: {filepath}")
                    
                processed_files.add(str(csv_file))
                
            except Exception as e:
                print(f"{csv_file}: {str(e)}")
        
        return all_contents, processed_files

if __name__ == "__main__":
    sql_reader = SQLFileReader(base_directory="~/redbench/redbench/")
    workloads_dir = "~/redbench/workloads"
    
    try:
        results, processed_csv_files = sql_reader.process_variability_csv_files(workloads_dir)
        

        
        for i, filepath in enumerate(list(sql_reader.file_cache.keys())[:5]):
            print(f"  {i+1}. {filepath}")
        
        for i, item in enumerate(results[:3]):
            content_preview = item['content'][:100] + "..." if len(item['content']) > 100 else item['content']
            print(f"    {content_preview}")
            
    except Exception as e:
        print(f"{str(e)}")