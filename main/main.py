import tkinter as tk
from app_main import HandDataCollectorApp

def main():
    root = tk.Tk()
    app = HandDataCollectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
