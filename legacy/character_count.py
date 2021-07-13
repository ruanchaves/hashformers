import pandas as pd  
import os  
import numpy as np 

def main():
    path = '../../datasets'
    result = []
    for path, subpaths, files in os.walk(path):
        for filename in files:
            if not filename.endswith('.tar'):
                with open(os.path.join(path, filename), 'r') as f:
                    data = f.read()
                    lines = data.split('\n')
                    len_lines = [len(x) for x in lines]
                    record = {
                        "filename": filename,
                        "lines": len_lines
                    }
                    result.append(record)
    result = pd.DataFrame(result)
    result = result.to_csv('count.csv')

if __name__ == '__main__':
    main()