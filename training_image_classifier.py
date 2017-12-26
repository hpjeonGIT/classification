##
## Reference: http://python-textbok.readthedocs.io/en/1.0/Introduction_to_GUI_Programming.html
#http://effbot.org/tkinterbook/entry.htm
from tkinter import Tk, Label, Button, Entry
from PIL import ImageTk, Image
import os
import time 
Nmax = 6 # this will make 0...Nmax sub-folders. 
# You may use 0 for non-classfiable objects


class ImageClassifier:
    def __init__(self, master):
        self.master = master
        master.title("Quick Training Image Classifier")
        master.geometry("500x500")
        master.configure(background='grey')
        #
        # Label
        self.label = Label(master, text="Essential tool for code monkeys")
        self.label.pack()
        #
        # making subfolders
        dir_name = []
        for i in range(Nmax+1):
            dir_name.append(str(i))
        
        CWD = './'
        for any in dir_name:
            if (not os.path.exists(CWD + any)):
                os.makedirs(CWD+any)
                print("made a folder named", CWD+any)
        #
        # Searching images
        self.f0_list = []
        for subds, dirs, files in os.walk(CWD):
            for f0 in files:
                ext = f0.split('.')[-1].lower()
                if (ext == 'png' or ext == 'gif' or ext == 'jpg' or 
                    ext =='jpeg' or ext == 'tif' or ext == 'tiff'):
                    self.f0_list.append(f0)
       #http://effbot.org/tkinterbook/photoimage.htm
       # img must be stored in the self
       #
       # Image
        self.count = 0
        self.img = ImageTk.PhotoImage(Image.open(self.f0_list[self.count]))
        self.image = Label(master, image = self.img)
        self.image.pack(side = "bottom", fill = "both", expand = "yes")
        self.f0txt = Label(master, text = self.f0_list[self.count])
        self.f0txt.pack(side="bottom",fill="both",expand="yes")
        #
        # Entry
        def Entercallback(event):
            self.entry.value  = self.entry.get()
            self.DoSomething(self.entry.value)

        self.entry = Entry(master)
        self.entry.insert(0, "")
        self.entry.bind('<Return>',Entercallback)    
        self.entry.bind('<KP_Enter>',Entercallback)    
        self.entry.pack()
        #
        # Quit
        self.quit_button = Button(master, text="Quit", command=master.quit)
        self.quit_button.pack()

    def DoSomething(self, value):
        if int(value) <= Nmax and int(value) >= 0:
            print(self.f0_list[self.count], " is classified as ", value)
            oldf0 = './'+self.f0_list[self.count]
            newf0 = './'+value+'/'+self.f0_list[self.count]
            os.rename(oldf0, newf0)
            self.image.destroy()
            self.f0txt.destroy()
            self.count += 1
            if (self.count == len(self.f0_list)):
                print("completed all images")
                self.master.quit()
            # New image
            self.img = ImageTk.PhotoImage(Image.open(self.f0_list[self.count]))
            self.image = Label(self.master, image = self.img)
            self.image.pack(side = "bottom", fill = "both", expand = "yes")
            self.f0txt = Label(self.master, text = self.f0_list[self.count])
            self.f0txt.pack(side="bottom",fill="both",expand="yes")
            self.entry.delete(0, "end")
        else:
            print("What are you doing?")

def main():
    ## GUI
    window = Tk()
    my_gui = ImageClassifier(window)
    window.mainloop()

if __name__ == '__main__':
    a_time = time.time()
    main()
    print("walll time = ", time.time() - a_time)

