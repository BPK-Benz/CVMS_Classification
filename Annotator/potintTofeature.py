import os, glob
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import calculation as calc

def read_anno(dataframe):
	labels = {}
	for index, row in dataframe.iterrows():
		if not row['filename'] in labels:
			labels[row['filename']] = [None] * 19
		labels[row['filename']][int(row['region_id'])] = np.array([
					eval(row['region_shape_attributes'])['cx'],
					eval(row['region_shape_attributes'])['cy'],		
				])
	return labels

if __name__ == "__main__":
    base = os.getcwd()
    target = base+'/../FeatureExtraction/'
    dataframe = pd.DataFrame()
    for s in sorted(glob.glob(base+'/../Point*/*.csv')):
        
        
            # set path
        src_path = s
        dst_path = target+s.split('/')[-1]
        print(s.split('/')[-1])

        # load labels
        labels = read_anno(pd.read_csv(src_path))

        # run through
        data = []
        for key in labels:
            p = labels[key]
            if any(v is None for v in p): 
                print(key, 'has a missing point.', [i for i in range(19) if p[i] is None])
            d = {
                'filename': key,

                # C2
                'C2 distance 0-4': calc.distance(p[0], p[4]),
                'C2 angle 0-2-4': calc.angle(p[0], p[2], p[4]), # find b angle
                'C2 angle 1-2-3': calc.angle(p[1], p[2], p[3]), # find b angle
                'C2 height 0-2-4': calc.concave_height(p[0], p[2], p[4]), # find vertex to focus point
                'C2 height 1-2-3': calc.concave_height(p[1], p[2], p[3]), # find vertex to focus point
                'C2 ratio height': calc.ratio(calc.concave_height(p[1], p[2], p[3]),calc.concave_height(p[0], p[2], p[4])),
                # 'C2 parabola base': calc.parabola([p[0], p[1], p[2], p[3], p[4]]), # the higher the narrow


                # C3
                'C3 distance 5-11': calc.distance(p[11],p[5]),
                'C3 distance 9-10': calc.distance(p[9], p[10]),
                'C3 ratio distance': calc.ratio(calc.distance(p[11],p[5]),calc.distance(p[9],p[10])),
                'C3 slope 10-11': calc.slope(p[10],p[11]),

                'C3 distance 5-9': calc.distance(p[5], p[9]),
                'C3 angle 5-7-9': calc.angle(p[5], p[7], p[9]), # find b angle
                'C3 angle 6-7-8': calc.angle(p[6], p[7], p[8]), # find b angle
                'C3 height 5-7-9': calc.concave_height(p[5], p[7], p[9]), # find vertex to focus point
                'C3 height 6-7-8': calc.concave_height(p[6], p[7], p[8]), # find vertex to focus point
                'C3 ratio height': calc.ratio(calc.concave_height(p[6], p[7], p[8]),calc.concave_height(p[5], p[7], p[9])),
                # 'C3 parabola base': calc.parabola([p[5], p[6], p[7], p[8], p[9]]), # the higher the narrow


                # C4
                'C4 distance 12-18': calc.distance(p[12],p[18]),
                'C4 distance 16-17': calc.distance(p[16], p[17]),
                'C4 ratio distance': calc.ratio(calc.distance(p[12],p[18]),calc.distance(p[16],p[17])),
                'C4 slope 16-17': calc.slope(p[16],p[17]),

                'C4 distance 12-16': calc.distance(p[12], p[16]),
                'C4 angle 12-14-16': calc.angle(p[12], p[14], p[16]), # find b angle
                'C4 angle 13-14-15': calc.angle(p[13], p[14], p[15]), # find b angle
                'C4 height 12-14-16': calc.concave_height(p[12], p[14], p[16]), # find vertex to focus point
                'C4 height 13-14-15': calc.concave_height(p[13], p[14], p[15]), # find vertex to focus point
                'C4 ratio height': calc.ratio(calc.concave_height(p[13], p[14], p[15]),calc.concave_height(p[12], p[14], p[16])),
                # 'C4 parabola base': calc.parabola([p[12], p[13], p[14], p[15], p[16]]), # the higher the narrow

            }
            data.append(d)

        # save
        df = pd.DataFrame(data)
        df.to_csv(dst_path)
        dataframe = dataframe.append(df)
    dataframe.reset_index(drop=True).to_csv(target+'CVMS.csv')
    
    