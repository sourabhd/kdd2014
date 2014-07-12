#!/bin/sh
#$-cwd
#$-N KDD
#$-j y
#$-o /nfs/bigeye/sdaptardar/kddcup2014/kdd2014/log.$JOB_ID.out
#$-e /nfs/bigeye/sdaptardar/kddcup2014/kdd2014/log.$JOB_ID.err
#$-M sdaptardar@cs.stonybrook.edu
#$-m ea
#$-l hostname=detection.cs.stonybrook.edu
#$-pe default 8
#$-R y
export PATH="/nfs/bigeye/sdaptardar/installs/venv/anaconda/bin:$PATH"
export PYTHONPATH="/nfs/bigeye/sdaptardar/installs/venv/anaconda/lib/python2.7/site-packages:$PYTHONPATH"
#export LD_LIBRARY_PATH=/opt/matlab_r2010b/bin/glnxa64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
export DISPLAY=localhost:11.0
echo "Starting job: $SGE_JOB_ID"
/nfs/bigeye/sdaptardar/installs/venv/anaconda/bin/ipython /nfs/bigeye/sdaptardar/kddcup2014/kdd2014/print_prob.py  
echo "Ending job: $SGE_JOB_ID"




