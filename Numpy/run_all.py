def run_all_functions():
    import numpy100
    for i in dir(numpy100):
        item = getattr(numpy100, i)
        if callable(item):
            try: item()
            except: pass

if __name__ == '__main__':
    run_all_functions()
