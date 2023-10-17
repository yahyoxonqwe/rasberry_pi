# Deploy [yolov8](https://github.com/ultralytics/ultralytics) tflite model in [Raspberry Pi](https://www.raspberrypi.com) with [streamlit](https://github.com/streamlit/streamlit) and connect  ipcamera
The Raspberry Pi is a type of single-board computer (SBC), meaning that its entire hardware set is placed on a single electronics board. There are no CPU sockets, memory slots or extension buses, such as Peripheral Component Interconnect Express, or PCIe, to swap out or add components.

The board of a Raspberry Pi contains a CPU; memory; LAN, USB and micro HDMI ports; and a slot for a micro SD card

![image](https://assets.raspberrypi.com/static/raspberry-pi-4-labelled-f5e5dcdf6a34223235f83261fa42d1e8.png)

# Installation
Clone the repository:
``` bash
git clone https://github.com/yahyoxonqwe/raspberry_pi.git
```
Change into the project directory:
``` bash
cd raspberry_pi
```
Install the required dependencies:
``` bash
pip install -r requirements.txt
```
Run stream.py with port(7777)
``` bash
streamlit run stream.py --server.port 7777
```
