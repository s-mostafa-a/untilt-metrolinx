# untilt-metrolinx

Filing pattern is simple.
Put all the point cloud data (.csv files) that you want to untilt in the directory: `./data/point_cloud/input`
Put all the label data (.txt files) that you want to untilt in the directory: `./data/labels/input`

Then, run the code `un_tilt_point_cloud.py` and `un_tilt_labels.py`. 

Your untilted files will be ready in `./data/point_cloud/output` and `./data/labels/output` respectively for point cloud and labels!

Note: My python version for this project is `3.6`. You cannot use visualization (that uses the package mayavi) in python 3.8 and above!
