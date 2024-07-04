from datetime import datetime, timedelta

def OUTPUT_MODEL_NAME(name:str, number_clases:int)->str:
    return f"model_{name}_{number_clases}_outputs.pth"

# Se asume que la lista no está vacía
def GET_TIME_HMS_FORMAT(time)->str:
    

    if isinstance(time, timedelta):
        days, seconds = time.days, time.seconds

        h = days * 24 + seconds // 3600
        m = (seconds % 3600) // 60
        s = (seconds % 60)    
        
        output = f"{h:02}:{m:02}:{s:02}"
    else: 
        output = time.strftime("%H:%M:%S")
    

    return output 

# Se asume que la lista no está vacía
def MEAN_TIMES(list_times: list)->timedelta:
    
    #number_times = len(list_times)    
    #sum_seconds = sum(map(lambda t: datetime.timestamp(t), list_times))
    #media = sum_seconds / number_times
    #return datetime.fromtimestamp(media)
    #return datetime.fromtimestamp( sum(map(lambda t: datetime.timestamp(t), list_times)) / len(list_times))
    
    # Son datetime.timedelta objects:
    return timedelta(seconds=sum(map(lambda t: t.total_seconds(), list_times)) / len(list_times))

def MULTIPLY_TIME(time, alpha:float)->timedelta:
    if isinstance(time, timedelta):
        return time * alpha
    else:
        return timedelta(seconds=datetime.timestamp(time) * alpha)
    