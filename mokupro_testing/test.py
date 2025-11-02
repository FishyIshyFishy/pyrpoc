import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r"C:\Users\ishaa\Box\(L2 Sensitive) zhan2017\Zhang lab data\Ishaan\MokuOscilloscopeSlot4Data_20251015_050602_Traces.csv")
time = pd.to_numeric(data['time'])
output = pd.to_numeric(data['A'])
num = pd.to_numeric(data['B'])
den = pd.to_numeric(data['C'])

plt.plot(time, output, 'k-', label='fpga division output')
plt.plot(time, num, 'b--', label='numerator')
plt.plot(time, den, 'r--', label='denominator')
plt.legend()
plt.show()