import freehead as fh
import time

athread = fh.ArduinoThread()
athread.start()
athread.started_running.wait()

for i in range(254):
    athread.write_uint8(i, 120, 254 -i, i)

time.sleep(6)    
athread.should_stop.set()
athread.join()