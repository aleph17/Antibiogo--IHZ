import json
import os

img_dir = '/home/muhammad-ali/Downloads/images'
jsonf = json.load(open('/home/muhammad-ali/Downloads/images.json'))
images = os.listdir(img_dir)

# correspondance = enumerate(images)
# for i in correspondance:
#     print(i)
correspondance = {}
# List all files in the directory
for key, value in jsonf['file'].items():
    filename = jsonf['file'][key]['fname']
    file_number = jsonf['file'][key]['fid']
    new_filename = file_number + '.jpg'
    jsonf['file'][key]['fname'] = new_filename

    correspondance[new_filename] = filename
    old_file = os.path.join(img_dir, filename)
    new_file = os.path.join(img_dir, new_filename)

    os.rename(old_file, new_file)
    print(f'Renamed: {old_file} -> {new_file}')

with open('/home/muhammad-ali/Downloads/images.json', 'w') as f:
    json.dump(jsonf, f)
with open('/home/muhammad-ali/Downloads/correspondance.json', 'w') as f:
    json.dump(correspondance, f)