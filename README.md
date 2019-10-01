Test parameters are controlled from parameters.py, and different world configurations can be selected using rover_domain.py and rover_setup.py. Current implementation utilizes a CCEA for training rover neural networks. All output files are placed in a directory called Output_Data (this includes world configuraiton files such as rover positions, POI positions, and POI values).

To run rover domain simulation using cython code run the following commands in order:
python3 setup.py build_ext --inplace
python3 main.py

To run visualizer, go to the Visualizer directory and use command: python3 visualizer.py
