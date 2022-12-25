import os

directorys = ['backVoice']
for directory in directorys:
    for filename in os.listdir(directory):
        if  '.mp3' in filename:
            f = os.path.join(directory, filename)
            newName = f.replace('.mp3', '.wav')
            print(f'name =  {f}')
            print(f'newName = {newName}')
            os.system(f"ffmpeg -i {f} {newName}")
            os.system(f"del {f}")

