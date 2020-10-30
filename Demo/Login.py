# import the library
from appJar import gui
import os
import sys
#import GUI_Main

# handle button events
def press(button):
    if button == "Cancel":
        app.stop()
    else:
        usr = app.getEntry("Username")
        pwd = app.getEntry("Password")
        print("User:", usr, "Pass:", pwd)
        if usr=='AB14' and pwd=='123':
            os.system('GUI_Main.py')
            sys.exit()                                
        else:
            app.errorBox("Error","Invalid Id or Password")


app = gui("Login Window", "400x200")
# create a GUI variable called app
app.setBg("Lightgreen")
app.setFont(18)
# add & configure widgets - widgets get a name, to help referencing them later

app.addLabel("title", "Welcome")
app.setLabelBg("title", "blue")
app.setLabelFg("title", "orange")
app.addLabelEntry("Username")
app.addLabelSecretEntry("Password")

# link the buttons to the function called press
app.addButtons(["Submit", "Cancel"], press)
app.setFocus("Username")
# start the GUI
app.go()

