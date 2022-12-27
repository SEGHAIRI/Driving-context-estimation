# Driving context predection

This repository contains the algorithmic implementation of driving context predection. The code is part of the hydrafusion algorithm which you can find in the paper titled "HydraFusion: Context-Aware Selective Sensor Fusion for Robust and Efficient Autonomous Vehicle Perception,".
This code is intended to be used with the RADIATE dataset available here: https://pro.hw.ac.uk/radiate/.

## **Model**

hydranet.py -- contains the part of Hydrafusion algorithm which determine the driving context.

stem.py -- defines the stem modules in HydraFusion.

gate.py -- contains the gating module implementations.


The stems are built using a ResNet-18 backbone. HydraFusion can be used with any image-based multi-modal dataset. In our evaluations we used two cameras, one radar sensor, and one lidar sensor as inputs to the model.

## **Train the model**

To train the model using deep gatting module, in config file you should set the gate as DeepGatingModule. If you want to use attention gatting you set the gate as AttentionGatingModule.
Then run the following command :

  ```
  python3 ./Train.py 
  ```
## **Test the model**

To test the model, run the following comman: :

  ```
  python3 ./Test.py 
  ```

## **Authors (Expleo)**

* **Maroua LADHARI** - *Maintainer*
* **Issam SEGHAIRI** - *Initial software version & Maintainer*



