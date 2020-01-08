import os
import pandas as pd
import numpy as np
import nibabel as nib

IMAGE_PATH = 'P:\\CMB\\Output'
GT_CSV = 'P:\\CMB\\final_output.csv'
OUTPUT_PATH =  'P:\\CMB\\GT\\'

csv_data = pd.read_csv(GT_CSV, delimiter=',')

adni_dirs = os.listdir(IMAGE_PATH)

for file in adni_dirs:
    images = os.listdir(os.path.join(IMAGE_PATH, file))

    for img in images:

        np_image = nib.load(filename=os.path.join(IMAGE_PATH, file, img)).get_data().astype(np.float32).squeeze()
        build_gt = np.zeros(np_image.shape)

        subject, date = str.split(img,sep='_')
        date, ext = os.path.splitext(date)
        subject = subject.strip('0')
        subject = int(subject)
        date = int(date)
        #print(subject, date)
        #print(len(subject))
        df = pd.DataFrame(csv_data.loc[(csv_data['RID'] == subject) & (csv_data['SCANDATE'] == date)])
        #df = pd.DataFrame(csv_data.loc[(csv_data['RID'] == subject) & (csv_data.loc[(csv_data['RID'] == subject)])
        #print(df['RASLOCATIONS'])
        if(df.empty == False):
            #print(df)
            print(img)
            loc = df['RASLOCATIONS']
            for cmb_pos in loc:
                #verts = np.array(cmb_pos)
                verts = str.split(cmb_pos, ' ')

                verts[0] = verts[0].strip('-')
                verts[1] = verts[1].strip('-')
                verts[2] = verts[2].strip('-')

                verts[0] = int(float(verts[0]))
                verts[1] = int(float(verts[1]))
                verts[2] = int(float(verts[2]))

                build_gt[verts[0], verts[1], verts[2]] = 1.0

            new_image = nib.Nifti1Image(build_gt, affine=np.eye(4))
            save_path = OUTPUT_PATH + img + ext
            nib.save(new_image, save_path)


