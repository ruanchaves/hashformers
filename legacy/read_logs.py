import pandas as pd  
import os 

def main():
    filenames = os.listdir()
    filenames = [ x for x in filenames if x.strip(',').endswith('log') ]
    file_df = []
    for fname in filenames:
        with open(fname,'r') as f:
            lines = f.read().split('\n')
            lines = [ x for x in lines if 'elapsed' in x ]
            record = { 'filename': fname, 'speed': lines}
            file_df.append(record)
    file_df = pd.DataFrame(file_df)
    file_df.to_csv('speed.csv')
        

if __name__ == '__main__':
    main()