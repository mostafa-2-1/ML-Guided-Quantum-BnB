import tkinter as tk
from tkinter import ttk
import pandas as pd

def show_csv_table(csv_path, title):
    df = pd.read_csv(csv_path)

    root = tk.Tk()
    root.title(title)
    root.geometry("1200x700")

    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    # Scrollbars
    x_scroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)
    y_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL)

    tree = ttk.Treeview(
        frame,
        columns=list(df.columns),
        show="headings",
        xscrollcommand=x_scroll.set,
        yscrollcommand=y_scroll.set
    )

    x_scroll.config(command=tree.xview)
    y_scroll.config(command=tree.yview)

    x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
    y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    tree.pack(fill=tk.BOTH, expand=True)

    # Define columns
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=140)

    # Insert rows
    for _, row in df.iterrows():
        values = []
        for v in row:
            if isinstance(v, float):
                values.append(f"{v:.4f}")
            else:
                values.append(v)
        tree.insert("", tk.END, values=values)

    root.mainloop()
