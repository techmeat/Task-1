import numpy as ny
import scipy as sy
import wget
import matplotlib.pyplot as plt
import math
import os
import xml.etree.ElementTree as ET
import scipy.constants as sycon
import scipy.special as sycp
dx = 1000000
url = 'https://jenyay.net/uploads/Student/Modelling/task_02.xml'
if (os.path.exists('task_02.xml')==0):
    input_data = wget.download(url)
    print(input_data)

tree = ET.parse('task_02.xml')
root = tree.getroot()
for elem in root.iter('variant'):
    if elem.get('number') == '6':
        D = float(elem.get('D'))
        fmin = float(elem.get('fmin'))
        fmax = float(elem.get('fmax'))
        

f = ny.arange(fmin, fmax, dx)


R = D / 2
lmbd = sycon.c / f 
k = 2 * math.pi / lmbd

def hn(n, x): return sycp.spherical_jn(n, x) + 1j * sycp.spherical_yn(n, x)
def bn(n, x): return (x * sycp.spherical_jn(n - 1, x) - n * sycp.spherical_jn(n, x)) / (x * hn(n - 1, x) - n * hn(n, x))
def an(n, x): return sycp.spherical_jn(n, x) / hn(n, x)

arr_sum = [((-1) ** n) * (n+0.5) * (bn(n, k * R) - an(n, k * R)) for n in range(1, 30)]
summa = ny.sum(arr_sum, axis=0)
rcs = (lmbd ** 2) / ny.pi * (ny.abs(summa) ** 2)



plt.plot(f / 10e6, rcs)
plt.xlabel("$f, МГц$")
plt.ylabel(r"$\sigma, м^2$")
plt.grid()
plt.show()







data = ET.Element('data')

ET.indent(data)
frequencydata = ET.SubElement(data, 'frequencydata')
lambdadata = ET.SubElement(data, 'lambdadata')
rcsdata = ET.SubElement(data, 'rcsdata')
ET.indent(data)
for i in range(len(f)):
    ET.indent(data)
    freq = ET.SubElement(frequencydata, 'f')
    freq.text = str(f[i])
    ET.indent(data)
    lamb =  ET.SubElement(lambdadata, 'lambda')
    lamb.text = str(lmbd[i])
    ET.indent(data)
    RCS = ET.SubElement(rcsdata, 'rcs')
    RCS.text = str(rcs[i])
    ET.indent(data)


if not os.path.isdir("results"):
     os.mkdir("results")
mydata = ET.ElementTree(data)
full_path = os.path.join('results','Data_2.xml')
with open (full_path,"wb") as f:
    mydata.write(f)




