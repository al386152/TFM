from datetime import datetime, timedelta

def OUTPUT_MODEL_NAME(name:str, number_clases:int)->str:
    return f"model_{name}_{number_clases}_outputs.pth"

# Se asume que la lista no está vacía
def GET_TIME_HMS_FORMAT(time, time_format="%H:%M:%S.%f")->str:
    
    # Si es timedelta, lo volvemos a transformar en un datetime y, de ahí, a un string con el formato deseado.
    if isinstance(time, timedelta):
        time = datetime.strptime(str(time), time_format)
    
    return time.strftime(time_format)

# Se asume que la lista no está vacía
def MEAN_TIMES(list_times: list)->timedelta:
    return timedelta(seconds=sum(map(lambda t: t.total_seconds(), list_times)) / len(list_times))

def MULTIPLY_TIME(time, alpha:float)->timedelta:
    if isinstance(time, timedelta):
        return time * alpha
    else:
        return timedelta(seconds=datetime.timestamp(time) * alpha)


def validate_dataset(dataset, logger):
    # Esta función se tiene que hacer solo si es el hilo principal.
    class_counts = {}
    for _, label in dataset:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    for class_idx, count in class_counts.items():
        logger.info(f"Class {dataset.classes[class_idx]} ({class_idx}): {count} samples")
