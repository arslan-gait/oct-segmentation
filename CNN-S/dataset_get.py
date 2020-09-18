from dependencies import *

class OCT_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.imgs8bit_list = list(sorted(os.listdir(os.path.join(root_dir, "8bit"))))
        self.black_targets_list = list(sorted(os.listdir(os.path.join(root_dir, "black"))))

    def __len__(self):
        return len(self.imgs8bit_list)

    def __getitem__(self, idx):
        img8bit_instance_path = os.path.join(self.root_dir, "8bit", self.imgs8bit_list[idx])
        black_target_instance_path = os.path.join(self.root_dir, "black", self.black_targets_list[idx])
        
        img8bit = Image.open(img8bit_instance_path)
        black_target = Image.open(black_target_instance_path)
        #
        img8bit = np.array(img8bit).T
        black_target = np.swapaxes(np.array(black_target), 0, 1)

        img8bit = cv2.resize(img8bit, dsize=(496, 523), interpolation=cv2.INTER_CUBIC)
        black_target = cv2.resize(black_target, dsize=(496, 523), interpolation=cv2.INTER_CUBIC)

        unique_labels = np.unique(black_target) #probably 255
        unique_label = unique_labels[1]
        target_result = np.zeros((img8bit.shape[0]*3))
        for i in range(3):
            black_target_2dim = black_target[:, :, i]
            val_255 = np.where(black_target_2dim == unique_label)
            target_result[val_255[0]*(i+1)] = val_255[1]
        if self.transform is not None:
            img8bit = self.transform(img8bit)
            #target_result = self.transform(target_result)
            target_result = torch.from_numpy(target_result)
        return img8bit, target_result

#copied from pytorch_official_tutorial
#do nothing but for now maybe later will use

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    #if train:
        #transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)




"""
path = os.path.dirname(os.path.abspath(__file__)) 

oct = OCT_dataset(path, get_transform(train=True))
list_len = []
#for i in range(len(oct)):
for i in range(1):
    #print(i)
    img8bit, target_result = oct[i]

print(img8bit.size(), target_result.size())
"""