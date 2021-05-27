from tkinter import filedialog
from tkinter import *
from tkinter import messagebox
import webbrowser, random
from PIL import ImageTk,Image
import os

def tableSp(a,p,r,f):
    global acs,ps,rs,fs
    acs=round(a,3)
    ps=round(p,3)
    rs=round(r,3)
    fs=round(f,3)

def tableF(a,p,r,f):
    global af,pf,rf,ff
    af=round(a,3)
    pf=round(p,3)
    rf=round(r,3)
    ff=round(f,3)

def outputTable():
    
    opT = Tk()
    opT.title("DNA Barcode Performance Metrics")
    opT.geometry("600x450")
    #set window color
    opT['bg']='bisque'
    Label(opT,background="bisque", font=("calibri","15"),width=10,text="").grid(row= 0, column=1, padx=100)

    Label(opT,background="bisque", font=("calibri","15","bold"),width=30,text="Metrics for Species Classification"
    ).grid(row= 1, column=1, columnspan = 2)
    Label(opT,background="bisque", font=("calibri","15"),width=10,text="Accuracy").grid(row= 2, column=1, padx=100)
    Label(opT,background="bisque", font=("calibri","15","italic"),width=5,text=acs).grid(row= 2, column=2, padx=100)
    Label(opT,background="bisque", font=("calibri","15"),width=10,text="Precision").grid(row= 3, column=1, padx=100)
    Label(opT,background="bisque", font=("calibri","15","italic"),width=5,text=ps).grid(row= 3, column=2, padx=100)
    Label(opT,background="bisque", font=("calibri","15"),width=10,text="Recall").grid(row= 4, column=1, padx=100)
    Label(opT,background="bisque", font=("calibri","15","italic"),width=5,text=rs).grid(row= 4, column=2, padx=100)
    Label(opT,background="bisque", font=("calibri","15"),width=10,text="F1").grid(row= 5, column=1, padx=100)
    Label(opT,background="bisque", font=("calibri","15","italic"),width=5,text=fs).grid(row= 5, column=2, padx=100)

    Label(opT,background="bisque", font=("calibri","15"),width=10,text="").grid(row= 6, column=1, padx=100)

    Label(opT,background="bisque", font=("calibri","15","bold"),width=30,text="Metrics for Family Classification"
    ).grid(row= 7, column=1, columnspan = 2)
    Label(opT,background="bisque", font=("calibri","15"),width=10,text="Accuracy").grid(row= 8, column=1, padx=100)
    Label(opT,background="bisque", font=("calibri","15","italic"),width=5,text=af).grid(row= 8, column=2, padx=100)
    Label(opT,background="bisque", font=("calibri","15"),width=10,text="Precision").grid(row= 9, column=1, padx=100)
    Label(opT,background="bisque", font=("calibri","15","italic"),width=5,text=pf).grid(row= 9, column=2, padx=100)
    Label(opT,background="bisque", font=("calibri","15"),width=10,text="Recall").grid(row= 10, column=1, padx=100)
    Label(opT,background="bisque", font=("calibri","15","italic"),width=5,text=rf).grid(row= 10, column=2, padx=100)
    Label(opT,background="bisque", font=("calibri","15"),width=10,text="F1").grid(row= 11, column=1, padx=100)
    Label(opT,background="bisque", font=("calibri","15","italic"),width=5,text=ff).grid(row= 11, column=2, padx=100)
    
    def on_closing2():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            opT.destroy()

    opT.protocol("WM_DELETE_WINDOW", on_closing2)
    
    opT.mainloop()


def sendop(species,family):
    global spec,fam,genus
    spec = (species.split())[0]
    genus = spec[0] + ". " + ((species.split())[1])
    fam = family


def outputs():

    op = Tk()
    op.title("DNA Barcode Output")
    op.geometry("1000x450")
    #set window color
    op['bg']='bisque'

    target_path_3 = os.path.join(os.path.dirname(__file__), '..\pics\%s.jpg' %fam)
    # print(target_path_3)
    img=ImageTk.PhotoImage(Image.open(target_path_3).resize((250, 250), Image.ANTIALIAS), master = op)   
    
    # setting image with the help of label
    Label(op, image = img).grid(row = 1, column = 1, rowspan = 3)

    Label(op,background="bisque", font=("calibri","15","bold"),width=10,text="Dataset").grid(row= 1, column=2)
    Label(op,background="bisque", font=("calibri","15"),width=20,text=fam).grid(row= 1, column=3)
    Label(op,background="bisque", font=("calibri","15","bold"),width=10,text="Genus").grid(row= 2, column=2)
    Label(op,background="bisque", font=("calibri","15","italic"),width=20,text=spec).grid(row= 2, column=3)
    Label(op,background="bisque", font=("calibri","15","bold"),width=10,text="Species").grid(row= 3, column=2)
    Label(op,background="bisque", font=("calibri","15","italic"),width=20,text=genus).grid(row= 3, column=3)
    Label(op,background="bisque", font=("calibri","15","bold"),width=10,text="Barcode").grid(row= 4, column=2,pady=10)
    
    target_path_4 = os.path.join(os.path.dirname(__file__), '..\src/barcode.png')
     

    if(fam == "Algae" or fam == "Plants"):
        img1=ImageTk.PhotoImage(Image.open(target_path_4).resize((600, 40), Image.ANTIALIAS), master = op)  
        # setting image with the help of label
    
    else:
        img1=ImageTk.PhotoImage(Image.open(target_path_4), master = op)  


    # setting image with the help of label
    Label(op, image = img1).grid(row = 5, column = 2, columnspan=2, pady=10)

  

    def on_closing1():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            op.destroy()

    op.protocol("WM_DELETE_WINDOW", on_closing1)


    outputTable()
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

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()


