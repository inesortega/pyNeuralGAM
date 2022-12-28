#!bin/bash

python main_ucoimbra.py -i ucoimbra-modbus -o results -d 0.05
python main_ucoimbra.py -i ucoimbra-tcp -o results -d 0.4
python main_ucoimbra.py -i ucoimbra-ping -o results -d 0.4

