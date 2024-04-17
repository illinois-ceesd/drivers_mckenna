import os
import shutil

# export LD_LIBRARY_PATH="/home/tulio/ParaView-5.11.2-MPI-Linux-Python3.9-x86_64/lib":$LD_LIBRARY_PATH
# export PYTHONPATH="/home/tulio/ParaView-5.11.2-MPI-Linux-Python3.9-x86_64/lib/python3.9/site-packages"
from paraview.simple import *

# state file to use
LoadState('dummy.pvsm')

# image resolution
xres=1920
yres=1080

# get info about number of time steps and number of layouts in state file
layouts=GetLayouts()
timeKeeper=GetTimeKeeper()
#timeSteps=timeKeeper.TimestepValues
#views=GetRenderViews()
animation=GetAnimationScene()

# create one directory for each layout (remove first if it already exists)
l=0
for layout in layouts:
    l=l+1
    dirname='layout_{:02d}'.format(l)
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)


t=0
#for time in timeSteps:
#    t=t+1
#    animation.AnimationTime=time
if True:
    l=0
    for layout in layouts:
        l=l+1
        filename = 'layout_{:02d}'.format(l) + '/image_{:06d}.png'.format(t)
        print('writing file: ', filename)
        SaveScreenshot(filename, layouts[layout], SaveAllViews=1, ImageResolution=[xres, yres], SeparatorWidth=0)

sys.exit(0)
