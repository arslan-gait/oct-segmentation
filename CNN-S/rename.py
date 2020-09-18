#done!
"""
import os

path = os.path.dirname(os.path.abspath(__file__)) 

img_8bit = list(sorted(os.listdir(os.path.join(path, "8bit"))))
img_black = list(sorted(os.listdir(os.path.join(path, "black"))))

print(img_8bit[1], "\n", len(img_8bit[1]))


new_img_8bit_list = [None] * len(img_8bit)
new_img_black_list = [None] * len(img_black)

for i in range(len(img_8bit)):
    index_slice = img_8bit[i].find("slice") + 6
    index_u = img_8bit[i].find("u") - 1
    len_of_slice = index_u - index_slice
    string_of_zeros = "_" + (2 - len_of_slice)*"0"
    new_img_8bit_list[i] = img_8bit[i][3:6] + string_of_zeros +img_8bit[i][13:index_u] + "_uint8.tif"
    new_img_black_list[i] = img_8bit[i][3:6] + string_of_zeros +img_8bit[i][13:index_u] + "_black.tif"
    os.rename(path + "/8bit/" + img_8bit[i], path + "/8bit/" + new_img_8bit_list[i])
    os.rename(path + "/black/" + img_black[i], path + "/black/" + new_img_black_list[i])


for j in range(2):
    for i in range(len(img_8bit[j])):
        if i < len(new_img_8bit_list[j]):
            char = new_img_8bit_list[j][i]
            char2 = new_img_black_list[j][i]
        else:
            char = "eee"
            char2 = "eee"
        print(img_8bit[j][i],char, char2, i)
"""