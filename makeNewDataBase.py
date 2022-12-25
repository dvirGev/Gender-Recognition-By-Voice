import csv
import os
import shutil
from pydub import AudioSegment

folders = ['regular', 'slow', 'fast', 'dirty']
maxNumber = 1500

lst = os.listdir('backVoice')
i=0
def dirty(file1):
    newName = f'D:/dirty/{file1[-17:]}'
    global i
    sound1 = AudioSegment.from_wav(file1)
    sound2 = AudioSegment.from_wav(f'backVoice/{lst[i%len(lst)]}')
    # mix sound2 with sound1, starting at 5000ms into sound1)
    output = sound1.overlay(sound2, position=0)
    # save the result
    output.export(newName, format="wav")
    i+=1
    return newName
def speed(name, speed):
    newName = ''
    if speed == 'fast':
        newName = f'D:/fast/{name[-17:]}'
        os.system(f'ffmpeg -i {name} -af "atempo=2" {newName}')
    else:
        newName = f'D:/slow/{name[-17:]}'
        os.system(f'ffmpeg -i {name} -af "atempo=0.5" {newName}')
    return newName
def creat(name, index):
    if folders[index%len(folders)] == 'regular':
        newName = f'D:/regular/{couple[0][-17:]}'
        shutil.copyfile(name,newName)
        return newName
    elif folders[index%len(folders)] == 'dirty':
        return dirty(name)
    else:
        return speed(name, folders[index%len(folders)])



if __name__ == '__main__':
    newData = []
    genderCount = {'male': 0, 'female':0}
    with open('balanced-all.csv' ,'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader,None)
            for couple in reader:
                # print(couple)
                # print(couple[0][-17:])
                print(f'male: {genderCount["male"]}')
                print(f'female: {genderCount["female"]}')
                if genderCount['male'] == 3400 and genderCount['female'] == 3400:   
                    break
                if genderCount[couple[1]] >= 3400:
                    continue
                if genderCount[couple[1]] <= 1500:
                    newName = creat(couple[0], genderCount[couple[1]])
                else:
                    newName = creat(couple[0], 0)
                newData.append([newName , couple[1]])
                genderCount[couple[1]]+=1

    with open('new_balanced-all.csv' ,'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(newData)