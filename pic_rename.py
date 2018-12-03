import os


# Get the names and store them in a list
path = "data/video/new/"
names = os.listdir(path)
print(len(names))

i = 0

for name in names:
    os.rename(os.path.join(path, name), os.path.join(path, 'video2_' + str(i)+'.mp4'))
    i = i + 1

