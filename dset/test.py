import h5py

with h5py.File('SC4001E0.h5', 'r') as f:
    data = f.file
    print(data.keys())
    for key in data.keys():
        print(key)
        if(len(data[key].shape) > 1):
            print(data[key].shape)
        else:
            print(data.get(key)[()])