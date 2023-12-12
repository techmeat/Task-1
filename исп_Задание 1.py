import math
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import os
def func(x):
    pi = math.pi
    e = math.e
    A = 1.34941
    return -0.0001*((abs(math.sin(x)*math.sin(A)*math.exp(abs(-((math.sqrt(x**2+A**2)/pi))+100))+1))**0.1)
if __name__ == '__main__':
    xmin = -10.0
    xmax = 10.0
    count = 200
    xlist = np.linspace(xmin,xmax, count)
    ylist = [func(x) for x in xlist]
    plt.plot(xlist,ylist)
    plt.show()


data = ET.Element('data')
for i in range(count):
    ET.indent(data)
    row = ET.SubElement(data, 'row')
    
    
    x = ET.SubElement(row, 'x')
    
    y = ET.SubElement(row, 'y')
    ET.indent(data)
    x.text = str(xlist[i])
    y.text = str(ylist[i])
    
    



if not os.path.isdir("results"):
     os.mkdir("results")
mydata = ET.ElementTree(data)
full_path = os.path.join('results/','Data.xml')
with open (full_path,"wb") as f:
    mydata.write(f)
 
