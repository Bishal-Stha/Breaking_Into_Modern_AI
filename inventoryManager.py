import customtkinter as ctk
from tkinter import messagebox
from enum import Enum

# ===== THEME SETUP =====
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT = "#7c6af7"
DANGER = "#e05252"
SUCCESS = "#4caf82"
WARN = "#e0a050"
SUBTEXT = "#888899"

# ===== ENUM =====
class InventoryType(Enum):
    WRITING_INSTRUMENT = "Writing Instrument"
    PAPER_PRODUCT = "Paper Product"
    FILING_ITEM = "Filing Item"
    DESK_SUPPLIES = "Desk Supplies"
    ART_SUPPLIES = "Art Supplies"
    SCHOOL_ESSENTIALS = "School Essentials"
    MISCELLANEOUS = "Miscellaneous"

# ===== INVENTORY CLASS =====
class Inventory:
    def __init__(self, name, item_type, quantity, price):
        self.__name = name
        self.__item_type = item_type
        self.__quantity = quantity
        self.__price = price

    def get_name(self): return self.__name
    def get_type(self): return self.__item_type
    def get_quantity(self): return self.__quantity
    def get_price(self): return self.__price

    def set_name(self, n): self.__name = n
    def set_type(self, t): self.__item_type = t
    def set_quantity(self, q): self.__quantity = q
    def set_price(self, p): self.__price = p

    def __str__(self):
        return f"{self.__name} | {self.__item_type.value} | Qty: {self.__quantity} | ${self.__price:.2f}"


# ===== MAIN APP =====
class InventoryApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Inventory Manager")
        self.geometry("900x660")
        self.minsize(860, 600)
        self.inventory_list = []
        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ── Header ──
        header = ctk.CTkFrame(self, corner_radius=0, fg_color=("#1e1e2e", "#1e1e2e"))
        header.grid(row=0, column=0, columnspan=2, sticky="ew")

        ctk.CTkLabel(
            header, text="📦  Inventory Manager",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#e8e6f0"
        ).pack(side="left", padx=24, pady=16)

        self.count_label = ctk.CTkLabel(
            header, text="0 items",
            font=ctk.CTkFont(size=12),
            text_color=SUBTEXT
        )
        self.count_label.pack(side="right", padx=24)

        # ── Left Panel (Form) ──
        left = ctk.CTkFrame(self, corner_radius=12, width=280)
        left.grid(row=1, column=0, sticky="nsew", padx=(16, 8), pady=16)
        left.grid_propagate(False)
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            left, text="ITEM DETAILS",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=ACCENT
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(20, 10))

        # Form fields
        fields = [
            ("Name", "e.g. Ballpoint Pen"),
            ("Quantity", "e.g. 50"),
            ("Price ($)", "e.g. 1.99"),
        ]
        self.entries = {}
        for i, (label, placeholder) in enumerate(fields):
            ctk.CTkLabel(
                left, text=label.upper(),
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=SUBTEXT
            ).grid(row=1 + i*2, column=0, sticky="w", padx=20, pady=(10, 2))

            entry = ctk.CTkEntry(
                left, placeholder_text=placeholder,
                height=38, corner_radius=8,
                border_color=ACCENT, border_width=1
            )
            entry.grid(row=2 + i*2, column=0, sticky="ew", padx=20, pady=(0, 2))
            self.entries[label] = entry

        # Category dropdown
        ctk.CTkLabel(
            left, text="CATEGORY",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=SUBTEXT
        ).grid(row=7, column=0, sticky="w", padx=20, pady=(10, 2))

        self.type_menu = ctk.CTkOptionMenu(
            left,
            values=[t.value for t in InventoryType],
            height=38, corner_radius=8,
            fg_color="#2b2b3b",
            button_color=ACCENT,
            button_hover_color="#6a58e0",
            dropdown_fg_color="#1e1e2e",
        )
        self.type_menu.grid(row=8, column=0, sticky="ew", padx=20, pady=(0, 10))

        # Divider
        ctk.CTkFrame(left, height=1, fg_color="#2e2e45").grid(
            row=9, column=0, sticky="ew", padx=20, pady=10
        )

        # Buttons
        btn_data = [
            ("＋  Add Item",    self.create_item,  ACCENT,   "#6a58e0"),
            ("✎  Update",       self.update_item,  SUCCESS,  "#3d9068"),
            ("✕  Delete",       self.delete_item,  DANGER,   "#c03030"),
            ("↺  Clear",        self.clear_fields, "#3a3a4a", "#4a4a5a"),
        ]
        for i, (text, cmd, fg, hover) in enumerate(btn_data):
            ctk.CTkButton(
                left, text=text, command=cmd,
                height=38, corner_radius=8,
                fg_color=fg, hover_color=hover,
                font=ctk.CTkFont(size=13, weight="bold")
            ).grid(row=10+i, column=0, sticky="ew", padx=20, pady=4)

        # ── Right Panel (List) ──
        right = ctk.CTkFrame(self, corner_radius=12)
        right.grid(row=1, column=1, sticky="nsew", padx=(8, 16), pady=16)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(
            right, text="INVENTORY LIST",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=ACCENT
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(20, 8))

        # Column header row
        col_frame = ctk.CTkFrame(right, fg_color="#1a1a28", corner_radius=6, height=32)
        col_frame.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 4))
        col_frame.grid_propagate(False)
        for col, anchor in [("Name", "w"), ("Category", "w"), ("Qty", "e"), ("Price", "e")]:
            ctk.CTkLabel(
                col_frame, text=col.upper(),
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=SUBTEXT
            ).pack(side="left", padx=16, pady=6, expand=True, anchor=anchor)

        # Scrollable list
        self.scroll_frame = ctk.CTkScrollableFrame(
            right, corner_radius=8, fg_color="#111120"
        )
        self.scroll_frame.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 16))
        self.scroll_frame.grid_columnconfigure(0, weight=1)

        # ── Status Bar ──
        self.status_var = ctk.StringVar(value="Ready — add an item to get started")
        status_bar = ctk.CTkFrame(self, corner_radius=0, height=32, fg_color="#181820")
        status_bar.grid(row=2, column=0, columnspan=2, sticky="ew")

        ctk.CTkLabel(
            status_bar, textvariable=self.status_var,
            font=ctk.CTkFont(size=11), text_color=SUBTEXT
        ).pack(side="left", padx=16, pady=6)

    # ===== HELPERS =====

    def get_val(self, key):
        return self.entries[key].get()

    def clear_fields(self):
        for entry in self.entries.values():
            entry.delete(0, "end")
        self.type_menu.set(list(InventoryType)[0].value)
        self.status_var.set("Fields cleared")

    def refresh_list(self):
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        for i, item in enumerate(self.inventory_list):
            row_color = "#16162a" if i % 2 == 0 else "#111120"
            row = ctk.CTkFrame(self.scroll_frame, fg_color=row_color, corner_radius=6, height=40)
            row.grid(row=i, column=0, sticky="ew", pady=2)
            row.grid_columnconfigure((0, 1, 2, 3), weight=1)
            row.grid_propagate(False)

            values = [
                item.get_name(),
                item.get_type().value,
                str(item.get_quantity()),
                f"${item.get_price():.2f}",
            ]
            colors = ["#e8e6f0", "#b0aec8", "#a0e8b0", "#f7c96a"]
            for j, (val, col) in enumerate(zip(values, colors)):
                ctk.CTkLabel(
                    row, text=val,
                    font=ctk.CTkFont(size=12),
                    text_color=col, anchor="w"
                ).grid(row=0, column=j, sticky="ew", padx=10, pady=8)

            # Click to select
            idx = i
            for child in row.winfo_children():
                child.bind("<Button-1>", lambda e, ix=idx: self.on_row_click(ix))
            row.bind("<Button-1>", lambda e, ix=idx: self.on_row_click(ix))

        n = len(self.inventory_list)
        self.count_label.configure(text=f"{n} item{'s' if n != 1 else ''}")

    def on_row_click(self, index):
        self.selected_index = index
        item = self.inventory_list[index]
        self.entries["Name"].delete(0, "end")
        self.entries["Name"].insert(0, item.get_name())
        self.entries["Quantity"].delete(0, "end")
        self.entries["Quantity"].insert(0, str(item.get_quantity()))
        self.entries["Price ($)"].delete(0, "end")
        self.entries["Price ($)"].insert(0, str(item.get_price()))
        self.type_menu.set(item.get_type().value)
        self.status_var.set(f"Selected: {item.get_name()}")

    # ===== CRUD =====

    def create_item(self):
        try:
            name = self.get_val("Name").strip()
            if not name:
                raise ValueError("Name is required")
            quantity = int(self.get_val("Quantity"))
            price = float(self.get_val("Price ($)"))
            item_type = InventoryType(self.type_menu.get())
            self.inventory_list.append(Inventory(name, item_type, quantity, price))
            self.refresh_list()
            self.status_var.set(f"✓ '{name}' added successfully")
            self.clear_fields()
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))

    def update_item(self):
        if not hasattr(self, "selected_index"):
            self.status_var.set("⚠ Click a row to select an item first")
            return
        try:
            item = self.inventory_list[self.selected_index]
            name = self.get_val("Name").strip()
            item.set_name(name)
            item.set_quantity(int(self.get_val("Quantity")))
            item.set_price(float(self.get_val("Price ($)")))
            item.set_type(InventoryType(self.type_menu.get()))
            self.refresh_list()
            self.status_var.set(f"✓ '{name}' updated")
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))

    def delete_item(self):
        if not hasattr(self, "selected_index"):
            self.status_var.set("⚠ Click a row to select an item first")
            return
        name = self.inventory_list[self.selected_index].get_name()
        del self.inventory_list[self.selected_index]
        del self.selected_index
        self.refresh_list()
        self.clear_fields()
        self.status_var.set(f"✓ '{name}' deleted")


# ===== RUN =====
if __name__ == "__main__":
    app = InventoryApp()
    app.mainloop()
