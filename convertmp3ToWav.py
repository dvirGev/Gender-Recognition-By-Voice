import os

directorys = ['D:/test_male2']
for directory in directorys:
    for filename in os.listdir(directory):
        if  '.flac' in filename:
            f = os.path.join(directory, filename)
            newName = f.replace('.flac', '.wav')
            print(f'name =  {f}')
            print(f'newName = {newName}')
            os.system(f"ffmpeg -i {f} {newName}")
# os.system(f"del {f}")

