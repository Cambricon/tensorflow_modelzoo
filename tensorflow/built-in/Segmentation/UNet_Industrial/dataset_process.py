import os
import csv
if __name__ == '__main__':
    files_output = ['train_list.csv','test_list.csv']
    data_dir = ['/Train','/Test']
    dataset_path = os.environ.get('DATA_DIR') + '/raw_images/private/'
    header = ["image_filepath","lbl_image_filepath","is_defective"]
    for (filename_output, datadir) in zip(files_output, data_dir):
        for i in range(1,11):
            f = open(os.path.join(dataset_path, 'Class'+str(i)+ datadir+'/Label/Labels.txt'), 'r')
            csvfile = open(dataset_path + 'Class'+str(i)+'/'+filename_output, 'w', newline='')
            writer=csv.writer(csvfile, delimiter=",")
            writer.writerow(header)
            f.readline()
            lines = f.readlines()
            csv_text = [""]*3
            for line in lines:
                text = line.split('\t')
                if text[4] == "0":
                    csv_text[1] = ""
                    csv_text[2] = "0"
                else:
                    csv_text[1] = text[4][:-1]
                    csv_text[2] = "1"
                csv_text[0] = text[2]
                writer.writerow(csv_text)
            f.close()
            csvfile.close()
