#!/bin/bash
conda activate grp_037
cd /data2/dse316/grp_037/PIDNet/tools
# Run first Python script
python3 custom.py --r ../leftImg8bit/demoVideo/stuttgart_02/

# Run second Python script
python3 video.py ../leftImg8bit/demoVideo/stuttgart_02/outputs