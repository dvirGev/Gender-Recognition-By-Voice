# from pydub import AudioSegment
# from pydub.playback import play

# song = AudioSegment.from_mp3("2X.mp3")
# play(song)


from pydub import AudioSegment
import os
import csv

lst = os.listdir('backVoice')
i=0
def con(file1):
    global i
    sound1 = AudioSegment.from_wav(file1)
    sound2 = AudioSegment.from_wav(f'backVoice/{lst[i%len(lst)]}')
    # mix sound2 with sound1, starting at 5000ms into sound1)
    output = sound1.overlay(sound2, position=0)
    # save the result
    output.export(f"data/dirty/{file1[-17:]}", format="wav")
    i+=1

maxNumber = 250
if __name__ == '__main__':
    newData = []
    genderCount = {'male': 0, 'female':0}
    with open('balanced-all.csv' ,'r', newline='') as csvfile:
           reader = csv.reader(csvfile)
           next(reader,None)
           for couple in reader:
                print(couple)
                if genderCount[couple[1]] >= maxNumber:
                    continue
                genderCount[couple[1]]+=1
                print(couple[0][-17:])
                name = f'data/dirty/{couple[0][-17:]}'
                con(couple[0])
                # os.system(f'ffmpeg -i {couple[0]} -af "atempo=0.5" {name}')
                newData.append([name , couple[1]])
                if genderCount['male'] == maxNumber and genderCount['female'] == maxNumber:   
                    break
    with open('balanced-all.csv' ,'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(newData)