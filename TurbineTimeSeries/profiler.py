from datetime import datetime


class Profiler:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def start(self, transformation, event_name):
        with open('log', 'a') as f:
            f.write(str(datetime.now())+','+transformation.type() + ',start,' + event_name + '\n')

    def end(self, transformation, event_name):
        with open('log', 'a') as f:
            f.write(str(datetime.now())+','+transformation.type() + ',end,' + event_name + '\n')
