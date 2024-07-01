import task2
import task3
import pandas as pd
import part1

def main(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    results=[]

    for index, row in df.iterrows():
        preprocessed_imgs = part1.preprocess_images([row['img1_path'],row['img2_path']])
        imgs_final = part1.detect_edges(preprocessed_imgs)
        imgs_final=part1.postprocess_images(imgs_final)

        
        d={"img1_path":row['img1_path'], "img2_path":row['img2_path']}
        num_components1, centroid1,leftmostpt1 = part1.count_connected_components(imgs_final[0])
        num_components2, centroid2,leftmostpt2 = part1.count_connected_components(imgs_final[1])

        mean1,var1=task2.inter_suture_distance(centroid1)
        mean2,var2=task2.inter_suture_distance(centroid2)

        height1 = imgs_final[0].shape[0]
        height2 = imgs_final[1].shape[0]

        var1=var1/(height1**2)
        var2=var2/(height2**2)

        if var1<var2:
            d["output_distance"]=1
        else:
            d["output_distance"]=2

        meanang1,varang1=task3.find_angle_mean_var(centroid1,leftmostpt1)
        meanang2,varang2=task3.find_angle_mean_var(centroid2,leftmostpt2)
        

        if varang1<varang2:
            d["output_angle"]=1
        else:
            d["output_angle"]=2
        results.append(d)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

