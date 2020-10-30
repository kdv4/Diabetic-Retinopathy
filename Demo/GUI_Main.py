from appJar import gui
import os
from shutil import copy
from test import Prediction
import glob


image_path=" "
img=" "
temp=" "
prev=" "
cpy_path="test/class/"

def checkStop():
    global app
    return app.yesNoBox("Confirm Exit", "Are you sure you want to exit the application?")

def press_info(win):
    app.showSubWindow(win)

def press_copy(win):
    app.showSubWindow(win)

def check_stage():
    global image_path
    if image_path==" ":
        app.errorBox("Error","Please Browse image first")
    else:
        x=Prediction(image_path)
        app.setLabel("stage", x)
    
def delete():
    folder = cpy_path
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(e)
        
def Browse():
    global image_path
    global temp
    if image_path==" ":
        x=app.openBox(title="Open image", dirName=None, fileTypes=[('images', '*.png'), ('images', '*.jpg'),('images', '*.jpeg')], asFile=False, parent=None)
        delete()
        copy(x,cpy_path)
        image_path=os.path.realpath(x)
        temp=x.split('/')[-1]
        app.startLabelFrame(temp,2,3)#y,x
        app.addImage("Image",image_path)
        app.shrinkImage("Image",2)
        app.stopLabelFrame()
    else:
        x=app.openBox(title="Open image", dirName=None, fileTypes=[('images', '*.png'), ('images', '*.jpg'),('images', '*.jpeg')], asFile=False, parent=None)
        delete()
        copy(x,cpy_path)
        image_path=os.path.realpath(x)
        app.setLabelFrameTitle(temp,x.split('/')[-1])
        app.reloadImage("Image",image_path)
        app.shrinkImage("Image",2)
        app.clearLabel("stage")


        
try:
    with gui("RemindMe","fullscreen",font={"size":19}) as app:
        #For main screen 
        app.setBg("Lightgreen")
        app.addEmptyLabel("stage",1,3)
        app.addLabel("title","Detection Of Diabetic Retinopathy!",0,2)
        app.addButtons(["Check Stage"],[check_stage],1,2)
        app.addButtons(["Import Image"],[Browse],1,1)
        
        app.addLink("All Rights reserved,AB14,© Copyright-2019", press_copy,4,2)
        app.addButtons(["Info"],[press_info],4,3)
        app.setStopFunction(checkStop)

        #This is for info window
        app.startSubWindow("Info",modal=True)
        app.addLabel("sT1","This Software is used to support the decision of doctor for identifying the level of diabetes")
        app.addLabel("sT2","To run a system: ")
        app.addLabel("sT3","1. Press Browse button to upload image")
        app.addLabel("sT4","2. Select Fundus image of retina whose Stage you want find")
        app.addLabel("sT5","3. Press Check Stage button to identify the stage of diabetes retinopathy")
        app.stopSubWindow()

        #For copyright window
        app.startSubWindow("All Rights reserved,AB14,© Copyright-2019",modal=True)
        app.addLabel("This software is reserved by AB14")
        app.stopSubWindow()
        
        app.go()
        
except:
    print("App stopped!")
