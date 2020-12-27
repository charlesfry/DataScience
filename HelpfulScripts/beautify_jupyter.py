import os
import getpass

startup_file = '''
import os\n
import pandas as pd\n
import numpy as np\n
import matplotlib.pyplot as plt\n
import seaborn as sns\n
from jupyterthemes import jtplot\n

jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)\n
'''

def main():
    user = getpass.getuser()
    path = f'C:\\Users\\{user}\\.ipython\\profile_default\\startup\\00_startup.py'
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w+') as f:
            for line in startup_file.split('\n'):
                print(line)
                f.write(f'{line}\n')
    os.system('pip install jupyterthemes')
    os.system('jt -t solarizedd -f fira -fs 115 -kl -N -cursc b')
    exec(open(path).read())
    print('\nJupyterThemes set')

if __name__ == '__main__':
    main()