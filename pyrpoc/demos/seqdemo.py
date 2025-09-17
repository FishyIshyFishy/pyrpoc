# Python demo for PicoHarp330 SEQ Mode

import time
import sys

# First we retrieve the call parameters
p1 = int(sys.argv[1]) 
p2 = int(sys.argv[2]) 

print("Python demo called with parameters p1=%d p2=%d" % (p1, p2) ) 

# The PicoHarp 330 software will always start a sequence with 
# p1 = 0 and increment it at each new call, up to the 
# specified stop index 

# Here you would put your code for e.g. setting a monochromator
# making use of p1 to e.g. encode the wavelength.
# You could do something like
#     wavelength = startwavelength + p1 * wavelengthstep
# and then set your monochromator accordingly.

# You can also return a value for recording as p3 in the PHU data file. 
# Just for demo purposes we return p1 + 1000.
# You can return any positive 32-bit integer, 
# for instance, following the example above, to indicate a wavelength. 
# Negative values are reserved for error codes and will cause the 
# measurement sequence to be aborted.
# Note that python may return 1 in case of syntax errors
# and it may return 2 when the script is not found.

returncode = p1 + 1000

print("Returncode will be %d" % returncode) 

#Just for debug/test purposes we keep the window visible for some time:
time.sleep(5) 

exit(returncode)
