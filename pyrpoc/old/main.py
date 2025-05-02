def main():
    import tkinter as tk
    from pyrpoc.old.mains.gui import GUI

    root = tk.Tk()
    app = GUI(root)
    root.mainloop()

if __name__=='__main__':
    main()