import re
import os

hashtags_folder = '/run/media/user/DADOS/NLP/datasets/hashtags'


def main():
    pattern = 'hashtags_(.*?).txt'
    hashtags = []
    for filename in os.listdir(hashtags_folder):
        path = os.path.join(hashtags_folder, filename)
        try:
            _ = re.search(pattern, path).group(1)
        except:
            continue
        
        with open(path,'r') as f:
            for line in f:
                hashtags.append(line)
    
    print(len(hashtags))
    print(len(set(hashtags)))
    assert(len(hashtags) == len(set(hashtags)))

if __name__ == '__main__':
    main()