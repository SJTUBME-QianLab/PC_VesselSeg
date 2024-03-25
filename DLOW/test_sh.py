import os

# generate code sample
os.system('python test_3D_slice.py --dataroot "datapath" --name V2D '
          '--model test --dataset_mode single --rand 0.1 --gpu 0 --results_dir '
          './data/perturbed_data'
          )
os.system('python test_3D_slice.py --dataroot "datapath" --name V2D '
          '--model test --dataset_mode single --rand 0.2 --gpu 0 --results_dir '
          './data/perturbed_data'
          )
os.system('python test_3D_slice.py --dataroot "datapath" --name V2D '
          '--model test --dataset_mode single --rand 0.3 --gpu 0 --results_dir '
          './data/perturbed_data'
          )
os.system('python test_3D_slice.py --dataroot "datapath" --name V2D '
          '--model test --dataset_mode single --rand 0.4 --gpu 0 --results_dir '
          './data/perturbed_data'
          )
os.system('python test_3D_slice.py --dataroot "datapath" --name V2D '
          '--model test --dataset_mode single --rand 0.5 --gpu 0 --results_dir '
          './data/perturbed_data'
          )


os.system('python test_3D_slice.py --dataroot "./data/data"--name V2A '
          '--model test --dataset_mode single --rand 0.1 --how_many 50 --gpu 0 --results_dir '
          './data/perturbed_data')
os.system('python test_3D_slice.py --dataroot "./data/data"--name V2A '
          '--model test --dataset_mode single --rand 0.2 --how_many 50 --gpu 0 --results_dir '
          './data/perturbed_data')
os.system('python test_3D_slice.py --dataroot "./data/data"--name V2A '
          '--model test --dataset_mode single --rand 0.3 --how_many 50 --gpu 0 --results_dir '
          './data/perturbed_data')
os.system('python test_3D_slice.py --dataroot "./data/data"--name V2A '
          '--model test --dataset_mode single --rand 0.4 --how_many 50 --gpu 0 --results_dir '
          './data/perturbed_data')
os.system('python test_3D_slice.py --dataroot "./data/data"--name V2A '
          '--model test --dataset_mode single --rand 0.5 --how_many 50 --gpu 0 --results_dir '
          './data/perturbed_data')