from read_utils import read_file, write_pickle
from cpu_test import  interferometer,_get_interferometer_on_fock_space_fwd, _get_interferometer_on_fock_space_bwd
import tkinter as tk
from tkinter import ttk
import tkinter.font
import pickle
import numpy as np



interferometer =  [
            [
                -0.11035524 + 0.43053175j,
                0.16672794 - 0.47819775j,
                0.01831264 - 0.07497556j,
                0.44214383 + 0.23308614j,
                0.49732734 - 0.20707815j,
            ],
            [
                -0.2226116 - 0.39735917j,
                0.30063104 - 0.55866103j,
                0.00329231 - 0.28009116j,
                0.23376629 - 0.22854039j,
                -0.43588261 + 0.12139052j,
            ],
            [
                0.24737463 - 0.24638855j,
                -0.31444021 - 0.28467995j,
                -0.16425914 - 0.33655956j,
                -0.26884137 + 0.68435951j,
                -0.07230938 - 0.10989756j,
            ],
            [
                0.34411568 + 0.00861778j,
                -0.26208576 - 0.04842002j,
                0.64971999 - 0.05882925j,
                0.29164646 + 0.0682574j,
                0.01229276 + 0.54314998j,
            ],
            [
                -0.59541546 - 0.01014536j,
                -0.28295784 + 0.10016806j,
                0.57604147 - 0.13381814j,
                -0.07509227 + 0.08557502j,
                -0.12237335 - 0.42143858j,
            ],
        ]
write_pickle("int.pkl", interferometer)

created_results = []
created_result_frames = []
created_inputs = []
window = tk.Tk()
container = ttk.Frame(window)
canvas = tk.Canvas(container)
scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
scrollbar_horz = ttk.Scrollbar(container, orient="horizontal", command=canvas.xview)
scrollable_frame = ttk.Frame(canvas)
window.title("Interferometer matrix calculator")
output_frame = tk.Frame(scrollable_frame, bg="skyblue")
output_canvas = tk.Canvas(output_frame, bg="skyblue")
input_frame = tk.Frame(scrollable_frame, bg="skyblue")
cutoff_frame = tk.Frame(input_frame, bg="skyblue")
d_frame = tk.Frame(input_frame, bg="skyblue")
gradient_frame = tk.Frame(input_frame, bg="skyblue")
interferometer_frame = tk.Frame(input_frame, bg="skyblue")

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.configure(xscrollcommand=scrollbar_horz.set)
container.pack(fill="both", expand=True)
scrollbar_horz.pack(side="bottom", fill="x")
scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)



scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

def show_upstream():
    if int(grad.get()) == 0:
          for inp in created_inputs:
            inp.destroy()
    else:
        l = tk.Label(gradient_frame, font=subtitle_font, background=label_bg, fg=label_fg,text="upstream: ")
        created_inputs.append(l)
        l.pack(fill="both", expand=True)
        e = tk.Entry(gradient_frame, font=subtitle_font,fg=label_fg, bg=entry_color)
        created_inputs.append(e)
        e.pack(fill="both", expand=True) 
def show_result_fields():
    bwd = grad.get()
    #TODO check
    cutoff = np.array([int(e1.get())], dtype=np.uint64)
    d = np.array([int(e2.get())], dtype=np.uint64)
    interferometer_path = e3.get()
    
    if bwd:
         upstream_path = created_inputs[1].get()
         interferometer = read_file(interferometer_path)
         upstream = read_file(upstream_path)
         resj, dimsj, helper_idxj, helper_sqrtj = _get_interferometer_on_fock_space_fwd(cutoff, d, interferometer)
         backward = _get_interferometer_on_fock_space_bwd(cutoff, interferometer,  d, resj, dimsj, helper_idxj, helper_sqrtj, upstream)
         print(resj)
         #print(backward)
         
    else:
         interferometer = read_file(interferometer_path)
         resj, dimsj, helper_idxj, helper_sqrtj = _get_interferometer_on_fock_space_fwd(cutoff, d, interferometer)
    #display result
    f_main = tk.Frame(output_frame, bg ="skyblue")
    created_result_frames.append(f_main)
    element_counter = 0
    result_label = tk.Label(f_main, bg=label_bg, font=subtitle_font ,text="Result:", fg=label_fg)
    result_label.grid(row=0, column=0, columnspan=len(dimsj), sticky='EW')
    for i, dim in enumerate(dimsj, start=0):
        
        f = tk.Frame(f_main, bg="skyblue")
        created_results.append(f)
        for row in range(dim):
             for col in range(dim):
                  disp = '{:.2f}'.format(resj[element_counter])
                  l = tk.Label(f, bg="skyblue", fg=label_fg, text=str(disp), borderwidth=1)
                  element_counter += 1
                  created_results.append(l)
                  l.grid(row=row, column=col, sticky='EW')
                  
        f.grid(row=1,column=i, sticky='EW')
    if bwd:
         grad_label = tk.Label(f_main, bg=label_bg, font=subtitle_font, text="Gradient:", fg=label_fg)
         grad_label.grid(row=2, column=0, columnspan=len(dimsj), sticky='EW')
         np.set_printoptions(precision=2)
         f2 = tk.Frame(f_main, bg="skyblue")
         for row in range(d[0]):
              for col in range(d[0]):
                  l = tk.Label(f2, bg="skyblue", fg=label_fg, text=str(backward[row*d + col]), borderwidth=1)
                  created_results.append(l)
                  l.grid(row=row, column=col, sticky='EW')
         f2.grid(row=3, column=0, columnspan=len(dimsj), sticky='EW')
    f_main.pack(fill="both", expand=True)
def delete_output():
    for res in created_results:
            res.destroy()
    for f in created_result_frames:
         f.destroy()
           


label_bg = "#CCC1B8"
label_fg = "#003664"
entry_color = "#DAE0E4"
title_font = tkinter.font.Font(family="Lato", size=24, weight="bold")
subtitle_font = tkinter.font.Font(family="Lato", size=18, weight="bold")
st_font = tkinter.font.Font(family="Lato", size=14)


greeting = tk.Label(scrollable_frame, font=title_font, text="Amazing interferometer calculator", background=label_bg, fg=label_fg).pack(fill="both", expand=True)
e1 = tk.Entry(cutoff_frame, font=subtitle_font,background=entry_color, fg=label_fg)
e2 = tk.Entry(d_frame, font=subtitle_font, background=entry_color, fg=label_fg)
e3 = tk.Entry(interferometer_frame, font=subtitle_font,background=entry_color, fg=label_fg)

tk.Label(cutoff_frame, font=subtitle_font, background=label_bg, fg=label_fg, text="cutoff: ").pack( fill="both", expand=True)
e1.pack( fill="both", expand=True)
tk.Label(d_frame, font=subtitle_font, background=label_bg, text="d: ", fg=label_fg).pack(fill="both", expand=True)
tk.Label(interferometer_frame,font=subtitle_font, background=label_bg, fg=label_fg,text="interferometer file name: ").pack(fill="both", expand=True)

e2.pack(fill="both", expand=True)
e3.pack(fill="both", expand=True)

grad = tk.IntVar()
bwd = ""
cutoff = ""
d = ""
interferometer_path = ""
upstream_path = ""

c1 = tk.Checkbutton(gradient_frame, font=subtitle_font, fg=label_fg, text='With gradient', variable=grad,onvalue=1, offvalue=0, command=show_upstream)
c1.pack(fill="both", expand=True)
input_frame.pack(fill="both", expand=True)
cutoff_frame.pack(padx=10, pady=(10,5), fill="both", expand=True)
d_frame.pack(padx=10, pady=5, fill="both", expand=True)
interferometer_frame.pack(padx=10, pady=5, fill="both", expand=True)
gradient_frame.pack(padx=10, pady=5, fill="both", expand=True)
tk.Button(input_frame, 
          text='Show output', command=show_result_fields).pack(
                                                       pady=5)
tk.Button(input_frame, 
          text='Delete output', command=delete_output).pack(
                                                       pady=5)
output_frame.pack(fill="both", expand=True)

window.mainloop()





