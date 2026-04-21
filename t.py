import h5py

with h5py.File('./dset/demo.h5', 'r') as f:
    print(f.keys())
    print(f['events'])
    print(f['hypnogram'][:].shape)
    print(f['signals'])
    print(f['signals']['eeg'].keys())
    print(f['signals']['eeg']['C4_M1'].shape)
    print(f['signals']['eog'].keys())
    print(f['signals']['eog']['EOG1'].shape)
    print(f['signals']['emg'].keys())
    print(f['signals']['emg']['ECG'].shape)
    print(f['signals']['emg']['EMG'].shape)