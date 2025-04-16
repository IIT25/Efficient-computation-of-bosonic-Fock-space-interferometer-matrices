from read_utils import read_file, write_pickle
from gpu_test import  interferometer,_get_interferometer_on_fock_space_fwd, _get_interferometer_on_fock_space_bwd, _get_interferometer_on_fock_space_xla
import tkinter as tk
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
write_pickle("u.pkl", interferometer)

created_results = []
created_inputs = []
window = tk.Tk()
output_frame = tk.Frame(window, bg="green")
input_frame = tk.Frame(window, bg="skyblue")
def show_upstream():
    if int(grad.get()) == 0:
          for inp in created_inputs:
            inp.destroy()
    else:
        l = tk.Label(input_frame, text="upstream: ")
        created_inputs.append(l)
        l.pack(fill="both", expand=True)
        e = tk.Entry(input_frame)
        created_inputs.append(e)
        e.pack(fill="both", expand=True) 
def show_entry_fields():
    l = tk.Label(output_frame, text="matrix ")
    created_results.append(l)
    l.pack(fill="both", expand=True, padx=5, pady=5)
    print("cutoff: %s\nd: %s\n ineterferometer file name: %s\n grad: %s" % (e1.get(), e2.get(), e3.get(), grad.get()))
    bwd = grad.get()
    cutoff = e1.get()
    d = e2.get()
    if bwd:
         interferometer = read_file(e3.get())
         upstream = read_file(created_inputs[1])
         forward = _get_interferometer_on_fock_space_fwd(cutoff, d, interferometer)
         backward = _get_interferometer_on_fock_space_bwd(forward, upstream)
         print(forward[0])
         print(backward)


          
def delete_output():
      for res in created_results:
            res.destroy()



greeting = tk.Label(text="Amazing interferometer calculator")
e1 = tk.Entry(input_frame)
e2 = tk.Entry(input_frame)
e3 = tk.Entry(input_frame)
greeting.pack()
tk.Label(input_frame, text="cutoff: ").pack(fill="both", expand=True)
tk.Label(input_frame, text="d: ").pack(fill="both", expand=True)
tk.Label(input_frame, text="interferometer file name: ").pack(fill="both", expand=True)

e1.pack(fill="both", expand=True)
e2.pack(fill="both", expand=True)
e3.pack(fill="both", expand=True)
tk.Button(input_frame, 
          text='Show output', command=show_entry_fields).pack(
                                                       pady=4)
tk.Button(input_frame, 
          text='Delete output', command=delete_output).pack(
                                                       pady=4)
grad = tk.IntVar()
c1 = tk.Checkbutton(input_frame, text='With gradient', variable=grad,onvalue=1, offvalue=0, command=show_upstream)
c1.pack(fill="both", expand=True)
input_frame.pack(padx=5, pady=5, fill="both", expand=True)

output_frame.pack(padx=5, pady=5, fill="both", expand=True)
window.mainloop()





