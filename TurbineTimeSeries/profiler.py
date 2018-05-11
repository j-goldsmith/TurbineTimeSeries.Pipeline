from datetime import datetime
import psutil

class Profiler:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def start(self, transformation, event_name):
        with open('time_log', 'a') as f:
            f.write(str(datetime.now())+','+transformation.type() + ',start,' + event_name + '\n')

        with open('mem_log', 'a') as f:
            f.write(str(psutil.virtual_memory().available)+','+transformation.type() + ',start,' + event_name + '\n')

        with open('disk_write_log', 'a') as f:
            f.write(str(psutil.disk_io_counters().write_time) + ',' + transformation.type() + ',start,' + event_name + '\n')

        with open('disk_read_log', 'a') as f:
            f.write(str(psutil.disk_io_counters().write_time) + ',' + transformation.type() + ',start,' + event_name + '\n')

        with open('cpu_log', 'a') as f:
            f.write(str(psutil.cpu_times().user) + ',' + transformation.type() + ',start,' + event_name + '\n')

    def end(self, transformation, event_name):
        with open('time_log', 'a') as f:
            f.write(str(datetime.now())+','+transformation.type() + ',end,' + event_name + '\n')

        with open('mem_log', 'a') as f:
            f.write(str(psutil.virtual_memory().available)+','+transformation.type() + ',end,' + event_name + '\n')

        with open('disk_write_log', 'a') as f:
            f.write(str(psutil.virtual_memory().available)+','+transformation.type() + ',end,' + event_name + '\n')

        with open('disk_read_log', 'a') as f:
            f.write(str(psutil.virtual_memory().available)+','+transformation.type() + ',end,' + event_name + '\n')

        with open('cpu_log', 'a') as f:
            f.write(str(psutil.cpu_times().user) + ',' + transformation.type() + ',end,' + event_name + '\n')