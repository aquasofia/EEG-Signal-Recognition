import os
from pathlib import Path


def file_name(ind):

    # Concatenate the filename from index
    return 'data/raw/' + str(ind) + '.txt'


def save_data(file, contents):

    # Check if file exists
    path = '/'.join(file.split('/')[:-1])
    if not Path(path).exists():
        os.mkdir(path)

    # Write to file
    with open(file, 'w') as file:
        for content in contents:
            file.write(content)
        file.close()


def main():

    contents = []
    with open('data/EP1.01.txt', 'r') as file_input:
        for ind, data_line in enumerate(file_input):
            contents.append(data_line)
            if ind % 50000 == 0 and ind != 0:
                print(ind, data_line)
                output_name = file_name(ind=ind)
                save_data(file=output_name,
                          contents=contents)
                contents = []


if __name__ == '__main__':
    main()
