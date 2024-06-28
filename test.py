from datetime import datetime

def get_training_time_HMS_format(time_start, time_end):
    intervalo = time_end - time_start

    days, seconds = intervalo.days, intervalo.seconds

    days = 1
    seconds = 3600*2 + 67

    h = days * 24 + seconds // 3600
    m = (seconds % 3600) // 60
    s = (seconds % 60)

    

    return f"{h:02}:{m:02}:{s:02}"


if __name__ == "__main__":
    t1 = datetime.now()
    s = get_training_time_HMS_format(t1, datetime.now())

    print(s)
    