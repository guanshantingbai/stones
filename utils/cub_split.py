"""
Split the original train.txt and test.txt to one-task and multi-task sets.
"""

train_file = "/media/data3/gdliu_data/Deep-Incremental-Image-Retrieval-main/data/train.txt"
test_file = "/media/data3/gdliu_data/Deep-Incremental-Image-Retrieval-main/data/test.txt"

origin_train = "./origin_train.txt"
new_train = "./new_train.txt"
origin_test = "./origin_test.txt"
new_test = "./new_test.txt"

new_train_s1 = "./new_train_s1.txt"
new_train_s2 = "./new_train_s2.txt"
new_train_s3 = "./new_train_s3.txt"
new_train_s4 = "./new_train_s4.txt"

new_test_s1 = "./new_test_s1.txt"
new_test_s2 = "./new_test_s2.txt"
new_test_s3 = "./new_test_s3.txt"
new_test_s4 = "./new_test_s4.txt"

train = open(train_file, 'r')
test = open(test_file, 'r')

origin_train = open(origin_train, 'w')
origin_test = open(origin_test, 'w')
new_train = open(new_train, 'w')
new_test = open(new_test, 'w')

new_train_s1 = open(new_train_s1, 'w')
new_train_s2 = open(new_train_s2, 'w')
new_train_s3 = open(new_train_s3, 'w')
new_train_s4 = open(new_train_s4, 'w')

new_test_s1 = open(new_test_s1, 'w')
new_test_s2 = open(new_test_s2, 'w')
new_test_s3 = open(new_test_s3, 'w')
new_test_s4 = open(new_test_s4, 'w')

train_info = train.readlines()
test_info = test.readlines()

for img_anon in train_info:
    [img, label] = img_anon.split(' ')
    label = int(label)
    if label <= 99:
        origin_train.write(img + ' ' + str(label) + '\n') 
    else:
        new_train.write(img + ' ' + str(label) + '\n')
    if label > 99 and label <= 124:
        new_train_s1.write(img + ' ' + str(label) + '\n')
    elif label > 124 and label <= 149:
        new_train_s2.write(img + ' ' + str(label) + '\n')
    elif label > 149 and label <= 174:
        new_train_s3.write(img + ' ' + str(label) + '\n')
    elif label > 174 and label <= 199:
        new_train_s4.write(img + ' ' + str(label) + '\n')
        
for img_anon in test_info:
    [img, label] = img_anon.split(' ')
    label = int(label)
    if label <= 99:
        origin_test.write(img + ' ' + str(label) + '\n') 
    else:
        new_test.write(img + ' ' + str(label) + '\n')
    if label > 99 and label <= 124:
        new_test_s1.write(img + ' ' + str(label) + '\n')
    elif label > 124 and label <= 149:
        new_test_s2.write(img + ' ' + str(label) + '\n')
    elif label > 149 and label <= 174:
        new_test_s3.write(img + ' ' + str(label) + '\n')
    elif label > 174 and label <= 199:
        new_test_s4.write(img + ' ' + str(label) + '\n')
 
train.close()
test.close()

origin_train.close()
origin_test.close()
new_train.close()
new_test.close()

new_train_s1.close()
new_train_s2.close()
new_train_s3.close()
new_train_s4.close()

new_test_s1.close()
new_test_s2.close()
new_test_s3.close()
new_test_s4.close()