'''
This is the script for running multiple seeds at a time.
If you want to test only one seed, you need to comment one line of code in each train.py and main.py. 

If you use linux to run this file, please change the command from python to python3
'''
import subprocess

for seed in range(21, 22):
    process1 = subprocess.Popen(['python', 'server.py'])

    process2 = subprocess.Popen(f'python train.py --seed={seed} --hospital=1', shell=True)
    process3 = subprocess.Popen(f'python train.py --seed={seed} --hospital=2', shell=True)

    process1.wait()
    process2.wait()
    process3.wait()

    process4 = subprocess.Popen(f'python main.py --seed={seed} --hospital=1', shell=True)
    process5 = subprocess.Popen(f'python main.py --seed={seed} --hospital=2', shell=True)

    process4.wait()
    process5.wait()
