from snAPI.Constants import LogLevel, MeasMode
from snAPI.Main import snAPI

sn = snAPI()
sn.getDevice()
sn.setLogLevel(LogLevel.DataFile, True)
sn.initDevice(MeasMode.T3)

sn.setLogLevel(logLevel=LogLevel.Config, onOff=True)
sn.loadIniConfig('')


num_chans = sn.deviceConfig['NumChans'] # pyright: ignore

