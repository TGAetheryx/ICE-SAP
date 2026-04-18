"""
SX1276 / RFM95W driver interface for ICE-SAP deployment.

Provides a hardware abstraction layer for LoRa communication.
On non-RPi platforms, falls back to simulation mode.
"""
import numpy as np
import time
from typing import Optional, Tuple


class SX1276Driver:
    """
    SX1276 LoRa driver (hardware or simulation).

    Hardware pins (RPi 4 GPIO):
      NSS  (SPI CS):  GPIO 8  (CE0)
      RESET:          GPIO 22
      DIO0:           GPIO 27
      SCK:            GPIO 11
      MOSI:           GPIO 10
      MISO:           GPIO 9
    """

    def __init__(self, simulation: bool = True, seed: int = 42):
        self.simulation = simulation
        self._rng = np.random.default_rng(seed)
        self._tx_count = 0
        self._rx_count = 0

        if not simulation:
            self._init_hardware()

    def _init_hardware(self):
        """Initialise SPI and configure SX1276 registers."""
        try:
            import spidev
            self._spi = spidev.SpiDev()
            self._spi.open(0, 0)
            self._spi.max_speed_hz = 5_000_000
            # LoRa mode: RegOpMode 0x81
            self._write_reg(0x01, 0x81)
        except ImportError:
            print("spidev not available — switching to simulation mode")
            self.simulation = True

    def _write_reg(self, addr: int, val: int):
        if not self.simulation:
            self._spi.xfer2([addr | 0x80, val])

    def _read_reg(self, addr: int) -> int:
        if not self.simulation:
            return self._spi.xfer2([addr & 0x7F, 0])[1]
        return 0

    def transmit(
        self,
        payload: bytes,
        spreading_factor: int = 12,
        bandwidth_khz: float = 125.0,
        tx_power_dbm: int = 14,
    ) -> Tuple[bool, float]:
        """
        Transmit payload via LoRa.

        Returns:
            success: bool.
            tx_time_s: Actual transmission time.
        """
        # Symbol rate
        sym_rate = (bandwidth_khz * 1000) / (2 ** spreading_factor)
        bits = len(payload) * 8 + 80   # payload + LoRa overhead
        tx_time = bits / (sym_rate * spreading_factor * 0.8)

        self._tx_count += 1

        if self.simulation:
            time.sleep(min(tx_time * 0.001, 0.001))  # fast simulation
            # Simulate occasional loss
            success = self._rng.random() > 0.02
            return success, tx_time

        # Hardware TX
        # Write payload to FIFO, set TX mode
        self._write_reg(0x0D, 0x00)    # FIFO TX base
        for b in payload[:255]:
            self._write_reg(0x00, b)
        self._write_reg(0x22, min(len(payload), 255))
        self._write_reg(0x01, 0x83)    # TX mode
        time.sleep(tx_time + 0.01)
        return True, tx_time

    @property
    def stats(self) -> dict:
        return {"tx_count": self._tx_count, "rx_count": self._rx_count}


if __name__ == "__main__":
    driver = SX1276Driver(simulation=True)
    payload = b"ICE-SAP boundary tile data"
    ok, t = driver.transmit(payload)
    print(f"TX: success={ok}, time={t:.2f}s")
    print(f"Stats: {driver.stats}")
