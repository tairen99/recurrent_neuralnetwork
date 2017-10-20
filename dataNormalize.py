

def dataNormalize(data, startDays, stopDays, std, mean):
    if std is None and mean is None:
        normalizeData = data[startDays:stopDays,:]
        mean = normalizeData.mean(axis=0)
        dataNomean = normalizeData - mean
        std = dataNomean.std(axis=0)
        data[startDays:stopDays, :] = dataNomean/std
        return data,mean,std
    else:
        # for the case when there are std, mean already
        normalizeData = data[startDays:stopDays,:]
        dataNomean = normalizeData - mean
        data[startDays:stopDays, :] = dataNomean/std
        return data,mean,std