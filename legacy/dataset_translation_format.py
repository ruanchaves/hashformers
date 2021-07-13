import sys
import os

def main():
    dataset_path = sys.argv[1]
    dataset_target = sys.argv[2]

    if os.path.isfile(dataset_target):
        sys.exit()

    with open(dataset_path,'r') as f:
        dataset = f.read().split('\n')
        new_file = []
        for line in dataset:
            new_line = line.strip().replace(" ", "") + ' = ' + line.strip()
            new_file.append(new_line)
        new_file = '\n'.join(new_file)
        with open(dataset_target,'w+') as f:
            print(new_file, file=f)


if __name__ == '__main__':
    main()