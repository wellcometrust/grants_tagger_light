class Benchmark:
    def __init__(self, filename):
        self.filename = filename
        self.metrics = {}

    def register(self, experiment_name, metric, time):
        if experiment_name not in self.metrics:
            self.metrics[experiment_name] = dict()
        self.metrics[experiment_name][metric] = time

    def to_csv(self):
        headers_written = False
        with open(self.filename, 'w') as f:
            for experiment_name, results in self.metrics.items():
                keys = results.keys()
                values = results.values()
                if not headers_written:
                    f.write(experiment_name)
                    headers_written = True
                    for k in keys:
                        f.write(";")
                        f.write(k)
                    f.write('\n')
                f.write(experiment_name)
                for v in values:
                    f.write(";")
                    f.write(v)
                f.write('\n')

