from tkinter import filedialog
from tkinter import *
from tkinter import messagebox
import webbrowser, random
from PIL import ImageTk,Image
import os

def sendop(species,family):
    global spec,fam,genus
    spec = (species.split())[0]
    genus = spec[0] + ". " + ((species.split())[1])
    fam = family


def outputs():

    op = Tk()
    op.title("DNA Barcode")
    op.geometry("600x650")
    #set window color
    op['bg']='bisque'
    Label(op,background="bisque", font=("calibri","15","bold"),width=10,text="Dataset").grid(row= 1, column=1, padx=70)
    Label(op,background="bisque", font=("calibri","15"),width=20,text=fam).grid(row= 1, column=2, padx=100)
    Label(op,background="bisque", font=("calibri","15","bold"),width=10,text="Genus").grid(row= 3, column=1, padx=70)
    Label(op,background="bisque", font=("calibri","15","italic"),width=20,text=spec).grid(row= 3, column=2, padx=100)
    Label(op,background="bisque", font=("calibri","15","bold"),width=10,text="Species").grid(row= 5, column=1, padx=70)
    Label(op,background="bisque", font=("calibri","15","italic"),width=20,text=genus).grid(row= 5, column=2, padx=100)
   
    

    op.mainloop()


def inputs():
     global dnainp,mlmodel
     dnainp = e1.get(1.0, "end-1c")
     e1.delete(1.0,END)
     mlmodel = variable.get()
     print(dnainp,mlmodel)
     root.destroy()




root = Tk()
root.title("DNA Barcode")
root.geometry("600x650")
 



#set width and height

canvas=Canvas(root,width=580,height=160)

# used dna.jpg from pics folder , took file directory of running file and traversed to pics folder
target_path_2 = os.path.join(os.path.dirname(__file__), '..\pics\dna.jpg')

image=ImageTk.PhotoImage(Image.open(target_path_2))

canvas.create_image(0,0,anchor=NW,image=image)
canvas.pack()


#set window color
root['bg']='bisque'


label =Label(text="Enter a DNA Barcode Sequene to Classify",font=("calibri","15"),width=60,activebackground="grey",bg="bisque")
label.pack(pady=20) 
e1 = Text(height=5, width=50,font=("calibri","12"))
e1.pack()
label =Label(text="Enter an ML model",font=("calibri","15"),width=60,activebackground="grey",bg="bisque")
label.pack(pady=10) 

variable = StringVar(root)
variable.set("Naive Bayes") # default value

w = OptionMenu(root, variable, "Naive Bayes", "SVM", "Random Forest","kNN")
w.configure(font=("calibri","15"),activebackground="grey",bg="white",activeforeground="white")
w["menu"].configure(font=("calibri","15"),activebackground="grey",bg="white")
w.pack()

button1 =Button(text='PREDICT',command=inputs,font=("calibri","15"),width=10,activebackground="grey",bg="white",activeforeground="white") 
button1.pack(pady=80) 


root.mainloop()


