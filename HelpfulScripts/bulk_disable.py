import subprocess
hand = 'C:\\Temp\disable.csv'

with open(hand, 'r') as file:
    for line in file:
        line = line.strip()
        command = f'Set-User -Identity "{line}" -RemotePowershellEnabled $False'

        p = subprocess.run(["powershell.exe", command])