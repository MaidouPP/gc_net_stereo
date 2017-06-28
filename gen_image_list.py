#-*-coding:utf-8-*-
import os
import random

def gen_images_list(data_dir='leftImg8bit/train/', label_dir='gtFine_trainId/train/'):
    sub_dirs_data = [f1 for f1 in os.listdir(data_dir) if os.path.isdir(os.path.abspath(os.path.join(data_dir,f1)))]
    sub_dirs_labels = [f2 for f2 in os.listdir(label_dir) if os.path.isdir(os.path.abspath(os.path.join(label_dir,f2)))]
    img_index = 0
    f_train = open('train.lst','w')
    train_files = []
    assert len(sub_dirs_data) == len(sub_dirs_labels)

    for sub_dir in sub_dirs_data:
        data_complete_dir = os.path.join(data_dir, sub_dir)
        label_complete_dir = os.path.join(label_dir, sub_dir)
        label_index = []
        label_files = []
        data_files = []

        for f in os.listdir(label_complete_dir):
            if os.path.isfile(os.path.abspath(os.path.join(label_complete_dir, f))) and 'labelIds' in f and 'gtFine' in f:
                f_lst = f.split('_')
                label_index.append(f_lst[1])
                label_files.append(os.path.join(label_complete_dir, f))
        label_files.sort()
        label_dir_files_len = len(label_files)
        
	# label_files = [os.path.join(label_sub_dir,f) for f in os.listdir(label_complete_dir) if os.path.isfile(os.path.abspath(os.path.join(label_complete_dir,f))) and 'labelIds' in f]

        for f in os.listdir(data_complete_dir):
            if os.path.isfile(os.path.abspath(os.path.join(data_complete_dir, f))):
                f_lst = f.split('_')
                if f_lst[1] in label_index:
                    data_files.append(os.path.join(data_complete_dir, f))
                    # data_files.append(os.path.join(data_complete_dir, f))
        data_dir_files_len = len(data_files)
        data_files.sort()
                    
        # data_files = [os.path.join(data_sub_dir,f) for f in os.listdir(data_complete_dir) if os.path.isfile(os.path.abspath(os.path.join(data_complete_dir,f)))]
        
        # those two lengths must be the same 
        assert label_dir_files_len==data_dir_files_len

        for data_file, label_file in zip(data_files, label_files):
            line = str(data_file) + '\t' + str(label_file) + '\n'
            f_train.write(line)
        print data_files, label_files
        
        # for index,file in enumerate(c_dir_files):
        #     line = str(img_index)+'\t'+img_label+'\t'+file+'\n'
        #     img_index+=1
        #     if index < c_dir_files_test_len:
        #         # f_test.write(line)
        #         test_files.append(line)
        #     else:
        #         train_files.append(line)

    f_train.close()

if __name__ == '__main__':
    gen_images_list()
