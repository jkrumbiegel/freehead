import freehead as fh
import time

pthread = fh.PupilThread()
pthread.start()
pthread.started_running.wait()


for i in range(3):
    time.sleep(10)
    pthread.reset_3d_eye_model()

pthread.should_stop.set()
pthread.join()