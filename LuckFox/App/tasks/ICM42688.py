import spidev
import time

class ICM42688:
    # Register addresses
    WHO_AM_I = 0x75
    PWR_MGMT_1 = 0x1F
    INT_STATUS = 0x13
    ACCEL_XOUT_H = 0x1E
    ACCEL_YOUT_H = 0x20
    ACCEL_ZOUT_H = 0x22
    GYRO_XOUT_H = 0x23
    GYRO_YOUT_H = 0x25
    GYRO_ZOUT_H = 0x27

    def __init__(self, bus=0, device=0, spi_speed_hz=24000000):
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = spi_speed_hz
        self.spi.mode = 0b00  # CPOL=0, CPHA=0

        # Wake up the sensor
        self.write_register(self.PWR_MGMT_1, 0x00)

    def write_register(self, reg_addr, data):
        # Reverse MOSI and MISO by swapping the order of bytes sent and received
        tx_buffer = [reg_addr, data]
        rx_buffer = self.spi.xfer2(tx_buffer[:])
        return rx_buffer[1]  # Return the received byte

    def read_register(self, reg_addr):
        # Reverse MOSI and MISO by sending a dummy byte to receive the actual data
        tx_buffer = [reg_addr | 0x80, 0x00]
        rx_buffer = self.spi.xfer2(tx_buffer[:])
        return rx_buffer[1]  # Return the received byte

    def read_accel_data(self):
        x = self.read_register(self.ACCEL_XOUT_H) << 8 | self.read_register(self.ACCEL_XOUT_H + 1)
        y = self.read_register(self.ACCEL_YOUT_H) << 8 | self.read_register(self.ACCEL_YOUT_H + 1)
        z = self.read_register(self.ACCEL_ZOUT_H) << 8 | self.read_register(self.ACCEL_ZOUT_H + 1)
        return {'x': x, 'y': y, 'z': z}

    def read_gyro_data(self):
        x = self.read_register(self.GYRO_XOUT_H) << 8 | self.read_register(self.GYRO_XOUT_H + 1)
        y = self.read_register(self.GYRO_YOUT_H) << 8 | self.read_register(self.GYRO_YOUT_H + 1)
        z = self.read_register(self.GYRO_ZOUT_H) << 8 | self.read_register(self.GYRO_ZOUT_H + 1)
        return {'x': x, 'y': y, 'z': z}

    def close(self):
        self.spi.close()

if __name__ == "__main__":
    icm = ICM42688(spi_speed_hz=24000000)
    print("WHO_AM_I:", hex(icm.read_register(ICM42688.WHO_AM_I)))

    try:
        while True:
            accel_data = icm.read_accel_data()
            gyro_data = icm.read_gyro_data()
            print(f"Accel X: {accel_data['x']}, Y: {accel_data['y']}, Z: {accel_data['z']}")
            print(f"Gyro X: {gyro_data['x']}, Y: {gyro_data['y']}, Z: {gyro_data['z']}")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        icm.close()



