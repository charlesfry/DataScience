import os
def main():
    requirements_path = 'C:\\Development\\requirements.txt'
    outfile = ''
    with open(requirements_path, 'r') as f:
        for line in f:
            outfile += line.strip() + '\n'

    with open('./requirements.txt', 'w+') as hand:
        hand.write(outfile)

    for line in outfile.split('\n'):
        line = line.split('==')[0]
        print(f'\tinstalling {line}')
        os.system(f'conda install {line} --y -c conda-forge')
        os.system(f'pip install {line} --use-feature=2020-resolver')

    os.system('conda update --all --y')

if __name__ == '__main__':
    main()