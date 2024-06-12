import glob

def find_file(date, time, action, extension):
    return glob.glob(f'videos/{action}_{date}_{time}.{extension}', recursive=True)