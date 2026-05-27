First, generate rough surfaces with
```shell
python generate_rough_surfaces.py params.toml
```

To run the simulation
```shell
mpiexec -np 4 python simulate.py params.toml
```
Every simulation creates a new subdirectory.

cd into a subdirectory and run
```shell
python level_set_approach.py params.toml
```
to complete the results.

Finally, run
```shell
python visualise.py
```
to visualise the results.
