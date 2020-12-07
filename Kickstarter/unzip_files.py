from zipfile import ZipFile
import os
from os import path

input_path = 'E:\KickStarter\zipped raw data'
output_path = 'E:\KickStarter\input'

for file in os.listdir(input_path):
    zip_path = path.join(input_path, file)
    unzip_path = path.join(output_path, file[:19])
    if not zip_path.endswith('.zip'): continue
    if not os.path.exists(unzip_path): os.mkdir(unzip_path)
    zip_file = ZipFile(zip_path)
    zip_file.extractall(unzip_path)