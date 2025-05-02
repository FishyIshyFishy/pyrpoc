# Created by Diego Alonso Alvarez on the 12th July 2020
#
# Copyright (c) 2020 Imperial College London
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

from ctypes import WinDLL, create_string_buffer
import os
import sys

path = r"C:\Users\Lab Admin\Documents\PythonStuff\pysrs\pysrs\instruments\prior_stage\PriorScientificSDK.dll"

if os.path.exists(path):
    SDKPrior = WinDLL(path)
else:
    raise RuntimeError("DLL could not be loaded.")

rx = create_string_buffer(1000)
realhw = True


def cmd(msg):
    print(msg)
    ret = SDKPrior.PriorScientificSDK_cmd(
        sessionID, create_string_buffer(msg.encode()), rx
    )
    if ret:
        print(f"Api error {ret}")
    else:
        print(f"46 OK {rx.value.decode()}")

    input("48 Press ENTER to continue...")
    return ret, rx.value.decode()


ret = SDKPrior.PriorScientificSDK_Initialise()
if ret:
    print(f"54 Error initialising {ret}")
    sys.exit()
else:
    print(f"57 Ok initialising {ret}")


ret = SDKPrior.PriorScientificSDK_Version(rx)
print(f"61 dll version api ret={ret}, version={rx.value.decode()}")


sessionID = SDKPrior.PriorScientificSDK_OpenNewSession()
if sessionID < 0:
    print(f"66 Error getting sessionID {ret}")
else:
    print(f"68 SessionID = {sessionID}")


ret = SDKPrior.PriorScientificSDK_cmd(
    sessionID, create_string_buffer(b"72 dll.apitest 33 goodresponse"), rx
)
print(f"74 api response {ret}, rx = {rx.value.decode()}")
input("75 Press ENTER to continue...")


ret = SDKPrior.PriorScientificSDK_cmd(
    sessionID, create_string_buffer(b"dll.apitest -300 stillgoodresponse"), rx
)
print(f"81 api response {ret}, rx = {rx.value.decode()}")
input("82 Press ENTER to continue...")


if realhw:
    print("Connecting...")
    # substitute 3 with your com port Id
    cmd("controller.connect 4")

    # test an illegal command
    cmd("controller.stage.position.getx")

    # get current XY position in default units of microns
    cmd("controller.stage.position.get")

    # re-define this current position as 1234,5678
    cmd("controller.stage.position.set 1234 5678")

    # check it worked
    cmd("controller.stage.position.get")

    # set it back to 0,0
    cmd("controller.stage.position.set 0 0")
    cmd("controller.stage.position.get")

    # start a move to a new position, normally you would poll
    # 'controller.stage.busy.get' until response = 0
    cmd("controller.stage.goto-position 1234 5678")

    # example velocity move of 10u/s in both x and y
    cmd("controller.stage.move-at-velocity 10 10")

    # see busy status
    cmd("controller.stage.busy.get")

    # stop velocity move
    cmd("controller.stage.move-at-velocity 0 0")

    # see busy status */
    cmd("controller.stage.busy.get")

    # see new position
    cmd("controller.stage.position.get")

    # disconnect cleanly from controller
    cmd("controller.disconnect")

else:
    input("Press ENTER to continue...")
