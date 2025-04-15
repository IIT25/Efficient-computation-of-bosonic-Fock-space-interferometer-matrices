from read_utils import write_pickle, read_pickle
#from gpu_test import  interferometer,_get_interferometer_on_fock_space_fwd, _get_interferometer_on_fock_space_bwd, _get_interferometer_on_fock_space_xla
import tkinter as tk
window = tk.Tk()
greeting = tk.Label(text="Amazing interferometer calculator")
tk.Button(window, text="Quit", command=window.destroy).grid(row=0)
e1 = tk.Entry(window)
e2 = tk.Entry(window)
e3 = tk.Entry(window)
greeting.grid(row=1)
tk.Label(window, text="cutoff: ").grid(row=2, column=0)
tk.Label(window, text="d: ").grid(row=3, column=0)
tk.Label(window, text="interferometer file name: ").grid(row=4, column=0)

e1.grid(row=2, column=1)
e2.grid(row=3, column=1)
e3.grid(row=4, column=1)

window.mainloop()

import pickle
import numpy as np
cutoff = 3
d = 5
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
write_pickle("test.pkl", interferometer)
with open("test.pkl", 'rb') as inp:
        parameters = pickle.load(inp)
print(parameters)
