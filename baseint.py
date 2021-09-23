"""
Basic Interface for the comb
"""
import tkinter as tk
import sys
import os
from artic import artic, loadBackGround, intBackGround, A4size
from PIL import ImageTk, Image

# millimeter to pixel
millimeter2pixel_ratio = 3.7795275591  # 3.77.. pixel for 1 mm, in dpi = 96


# main App
class baseint(tk.Frame):
    def __init__(self, master, dpi):
        super().__init__(master)
        self.master = master

        self.a4size_in_px = tuple(map(lambda x: int(dpi / 96 * millimeter2pixel_ratio * x), A4size))

        self.atc = artic()
        self.bk = loadBackGround()
        self.canv = tk.Canvas(master, width=400, height=600)
        self.canv.pack(side="right")

        self.relw_scale = tk.Scale(master, from_=0.7, to=1., length=200, resolution=0.01, orient=tk.HORIZONTAL
                                   , label="Relative main width")
        self.relw_scale.set(0.9)
        self.relw_scale.pack()

        self.relh_scale = tk.Scale(master, from_=0.7, to=1., length=200, resolution=0.01, orient=tk.HORIZONTAL
                                   , label="Relative main height")
        self.relh_scale.set(0.85)
        self.relh_scale.pack()

        self.dw_scale = tk.Scale(master, from_=-10, to=5, length=200, resolution=1, orient=tk.HORIZONTAL
                                 , label="Char spacing")
        self.dw_scale.set(-5)
        self.dw_scale.pack()

        self.dh_scale = tk.Scale(master, from_=-10, to=5, length=200, resolution=1, orient=tk.HORIZONTAL
                                 , label="Line spacing")
        self.dh_scale.set(-5)
        self.dh_scale.pack()

        self.x_scale = tk.Scale(master, from_=0., to=0.3, length=200, resolution=0.01, orient=tk.HORIZONTAL,
                                label="Relative x-offset")
        self.x_scale.set(0.12)
        self.x_scale.pack()

        self.y_scale = tk.Scale(master, from_=0., to=0.3, length=200, resolution=0.01, orient=tk.HORIZONTAL,
                                label="Relative y-offset")
        self.y_scale.set(0.15)
        self.y_scale.pack()

        self.mainb = tk.Button(master, text="Preview", command=self.rendPage)
        self.mainb.pack()
        self.mainb = tk.Button(master, text="Make!", command=self.gen)
        self.mainb.pack()

    def rendPage(self, relw=0.7, relh=0.8, dw=-5, dh=-5, x=0.12, y=0.15):
        # print("calling")
        relw = self.relw_scale.get()
        relh = self.relh_scale.get()
        dw = self.dw_scale.get()
        dh = self.dh_scale.get()
        x = self.x_scale.get()
        y = self.y_scale.get()

        a4w, a4h = self.a4size_in_px
        page = self.atc.makePage((int(relw * a4w), int(relh * a4h)), (dw, dh))
        res = intBackGround(page, self.bk, (x, y))
        res = res.resize((res.width // 2, res.height // 2))

        # sign-in res as class attribute to avoiding the gc before change canvas image
        self.res = ImageTk.PhotoImage(res)
        self.canv.create_image(0, 10, image=self.res, anchor="nw")

    def gen(self):
        relw = self.relw_scale.get()
        relh = self.relh_scale.get()
        dw = self.dw_scale.get()
        dh = self.dh_scale.get()
        x = self.x_scale.get()
        y = self.y_scale.get()
        a4w, a4h = self.a4size_in_px
        idp = 0
        for pg in self.atc.make((int(relw * a4w), int(relh * a4h)), (dw, dh)):
            idp += 1
            res = intBackGround(pg, self.bk, (x, y))
            res.save("./test/page_%.3d.png" % idp)









# Adjustable Parameters: dw, dh, x, y
# relw = 0.7
# relh = 0.8
# dw = -5  # interval length between characters
# dh = -5  # interval height between lines
# x = 0.12  # left-top position-x of content relative to background
# y = 0.15  # left-top position-y of content relative to background

if __name__ == "__main__":
    window = tk.Tk()
    window.title("test")
    window.geometry("800x600")
    dpi = window.winfo_fpixels('1i')  # get screen dpi
    main = baseint(window, dpi)
    # main.rendPage()
    main.mainloop()
