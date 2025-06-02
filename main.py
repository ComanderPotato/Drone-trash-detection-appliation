import tkinter as tk
from Application import TrashDetectorApp
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1920x1080")
    app = TrashDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
