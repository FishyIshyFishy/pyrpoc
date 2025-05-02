import concurrent.futures
from zaber_motion import Units
from zaber_motion.ascii import Connection

class ZaberStage:
    def __init__(self, port='3'):
        self.port = 'COM' + str(port)
        self.connection = None
        self.device = None
        self.axis = None

    def connect(self, timeout=10):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._connect)
            future.result(timeout=timeout)

    def _connect(self):
        if self.connection is not None:
            return
        self.connection = Connection.open_serial_port(self.port)
        self.connection.enable_alerts()
        devices = self.connection.detect_devices()
        if not devices:
            raise RuntimeError("No Zaber devices found.")
        self.device = devices[0]
        self.axis = self.device.get_axis(1)
        if not self.axis.is_homed():
            print("Homing the stage...")
            self.axis.home()

    def move_absolute_um(self, position_um):
        if self.axis is None:
            self.connect()
        position_mm = position_um * 1e-3
        self.axis.move_absolute(position_mm, Units.LENGTH_MILLIMETRES)
        self.axis.wait_until_idle()

    def is_connected(self):
        return (self.connection is not None)

    def disconnect(self):
        if self.connection:
            self.connection.close()
            self.connection = None
