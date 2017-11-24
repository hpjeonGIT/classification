# classification
Sample code motivated by the work of Agnes. Using Keras, TF, Horovod, and openmpi

1. Classifying the images of Miku, Rin, and Teto
  
  1.1 Collect images from bing or google image search
  
  1.2 Data preparation at train and val folders
  
  1.3 mpirun -n 4 -mca pml ob1 python3 step1_hvd.py
      This yields initial model and weights.
  
  1.4 mpirun -n 4 -mca pml ob1 python3 step2_hvd.py
      Accelerating training by using the weights from 1.3
  
  1.5 python3 classify.py --image __given_image.py
