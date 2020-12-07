from zipfile import ZipFile
import os
from os import path

def main(input_path, output_path):

    assert os.path.exists(input_path), f'{input_path} does not exist. Please check your input path and try again.'

    for file in os.listdir(input_path):
        print(f'Unzipping {file}...')
        zip_path = path.join(input_path, file)
        unzip_path = path.join(output_path, file)
        if not zip_path.endswith('.zip'): continue
        if not os.path.exists(unzip_path): os.mkdir(unzip_path)
        zip_file = ZipFile(zip_path)
        zip_file.extractall(unzip_path)

if __name__ == '__main__':
    input_path = input('Type the full or relative path of the directory that contains the .zip files that you wish to extract.\n')
    output_path = input('type the full or relative path that you would like to extract the files to\n')

    catch = input(f'Extracting from:\n{input_path}\n'
                  f'Extracting to:\n{output_path}\n'
                  f'Are you sure you want to extract zipped files (y/[n])?').lower()

    if catch not in ['y','yes']:
        print('User did not type \'y\', exiting program.')
        quit()

    main(input_path, output_path)